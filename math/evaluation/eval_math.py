#!/usr/bin/env python3
"""
GDPO math Track T2 eval: 5 benchmarks x pass@1@16 + Exceed.

Alignment:
- GDPO paper 4.2: AIME-24 / AMC(2022+2023) / MATH / Minerva / OlympiadBench,
  vLLM backend, temperature 0.6, top_p 0.95, max_response 32k, 16 samples/question,
  report average pass@1 and Exceed (fraction of responses over 4000 tokens).
- MATH subset = MATH-500 (evidence: DeepScaleR official rllm repo,
  rllm/data/preprocess/math/math.ipynb loads HuggingFaceH4/MATH-500 -> math500.json).
  High-confidence inference, not the authors' eval manifest itself.

Question counts verified against reviewer Opus-A9 expectations (all matched):
  MATH-500       HuggingFaceH4/MATH-500                          500
  AIME24         HuggingFaceH4/aime_2024                          30
  AMC            AI-MO/aimo-validation-amc (AMC12 22/23 A+B)      83
  Minerva        math-ai/minervamath                             272
  OlympiadBench  Hothan/OlympiadBench :: OE_TO_maths_en_COMP     674
  total 1559 questions x 16 samples = 24,944 generations

Metric definitions (reviewers GPT-A13 / Opus-A9):
- avg pass@1 = (1/Q) sum_q (1/16) sum_s 1[correct]  -- equal weight per question,
  NOT the pass@16 "at least one success" combinatorial formula
- Exceed = #{(q,s): len(token_ids) > 4000} / (16Q)  -- strict >4000, complement of
  the training reward's <=4000
- Exceed counts vLLM-returned token_ids; never re-tokenize decoded text
- 32k-truncated samples count toward Exceed and are scored incorrect
- eval prompt must match training exactly (no system msg, instruction in user turn,
  template pre-fills <think>)

Usage:
  python eval_math.py --ckpt <path> --out <json>            # all 5
  python eval_math.py --ckpt <path> --bench math500 --n 4   # quick check
"""
import argparse
import json
import os
from pathlib import Path
import sys

# Must stay in sync with preprocess_deepscaler.py
INSTRUCTION_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

BENCHES = {
    "math500":  dict(repo="HuggingFaceH4/MATH-500",    cfg=None,
                     split="test",  q="problem",  a="answer",     n_expect=500),
    "aime24":   dict(repo="HuggingFaceH4/aime_2024",   cfg=None,
                     split="train", q="problem",  a="answer",     n_expect=30),
    "amc":      dict(repo="AI-MO/aimo-validation-amc", cfg=None,
                     split="train", q="problem",  a="answer",     n_expect=83),
    "minerva":  dict(repo="math-ai/minervamath",       cfg=None,
                     split="test",  q="question", a="answer",     n_expect=272),
    "olympiad": dict(repo="Hothan/OlympiadBench",      cfg="OE_TO_maths_en_COMP",
                     split="train", q="question", a="final_answer", n_expect=674),
}


def load_bench(name, raw=False):
    """raw=True returns the ground truth without rendering normalization."""
    from datasets import load_dataset
    from answer_numeq import unwrap_math
    b = BENCHES[name]
    ds = load_dataset(b["repo"], b["cfg"], split=b["split"]) if b["cfg"] \
        else load_dataset(b["repo"], split=b["split"])
    if len(ds) != b["n_expect"]:
        print("[warn] %s: n=%d but expected %d - dataset may have changed; "
              "numbers will not be comparable." % (name, len(ds), b["n_expect"]))
    items = []
    for ex in ds:
        gt = ex[b["a"]]
        if isinstance(gt, list):          # OlympiadBench final_answer is a list
            gt = gt[0] if gt else ""
        if isinstance(gt, float) and gt.is_integer():
            # AMC answers are stored as floats (142.0); render integral values as integers
            # so the grader, which is shared with the training reward, is left unchanged.
            gt = int(gt)
        gt = str(gt)
        if not raw:
            # OlympiadBench answers are wrapped in LaTeX math delimiters ('$2^{1009}$');
            # strip the delimiters, again without touching the grader.
            gt = unwrap_math(gt)
        items.append({"problem": str(ex[b["q"]]), "gt": gt})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench", default="all")
    ap.add_argument("--n", type=int, default=16, help="samples per question (paper: 16)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--length_limit", type=int, default=4000, help="Exceed threshold")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default="auto",
                    help="Rollout jsonl path; 'auto' derives it from --out; '' or 'none' disables saving.")
    args = ap.parse_args()
    if args.dump in ("", "none", "no"):
        args.dump = None

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import math_grader as math_scorer
    from answer_numeq import numeq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import statistics as st

    names = list(BENCHES) if args.bench == "all" else args.bench.split(",")
    tok = AutoTokenizer.from_pretrained(args.ckpt)

    llm = LLM(model=args.ckpt, tensor_parallel_size=args.tp, dtype="bfloat16",
              max_model_len=args.max_tokens + 2048, gpu_memory_utilization=0.85,
              trust_remote_code=True, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)

    results = {}
    for name in names:
        items = load_bench(name)
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": it["problem"].strip() + INSTRUCTION_SUFFIX}],
                tokenize=False, add_generation_prompt=True)
            for it in items
        ]
        outs = llm.generate(prompts, sp)

        # Save every response (text, token count, scores) so grading can be redone offline.
        dump_f = None
        if args.dump:
            dump_path = args.dump if args.dump != "auto" else \
                os.path.splitext(args.out or "eval")[0] + "_%s_rollouts.jsonl" % name
            dump_f = open(dump_path, "w")
            print("[dump] rollouts -> %s" % dump_path)

        per_q_acc, per_q_acc_ne, tok_lens, n_exceed, n_total, n_resc, n_joint = [], [], [], 0, 0, 0, 0
        for qi, (it, o) in enumerate(zip(items, outs)):
            hits = hits_ne = 0
            for si, comp in enumerate(o.outputs):
                n_total += 1
                ntok = len(comp.token_ids)          # vLLM token count, incl. EOS
                tok_lens.append(ntok)
                if ntok > args.length_limit:
                    n_exceed += 1
                # 32k-truncated -> no boxed -> scored incorrect (no special-casing)
                sc = int(bool(math_scorer.compute_score(comp.text, it["gt"])))
                # Numeric-equivalence fallback, only tried when the strict grader says 0
                # (e.g. '025' vs 25, '4.5e33'); both strict and fallback accuracy are reported.
                ne = sc or int(numeq(comp.text, it["gt"]))
                if ne and not sc:
                    n_resc += 1
                hits += sc
                hits_ne += ne
                n_joint += int(bool(ne) and ntok <= args.length_limit)
                if dump_f:
                    dump_f.write(json.dumps({
                        "q": qi, "s": si, "gt": it["gt"], "score": sc, "score_numeq": ne,
                        "ntok": ntok, "finish": comp.finish_reason,
                        "text": comp.text,
                    }, ensure_ascii=False) + "\n")
            per_q_acc.append(hits / len(o.outputs))
            per_q_acc_ne.append(hits_ne / len(o.outputs))

        if dump_f:
            dump_f.close()

        tok_lens.sort()

        def pct(p):
            return tok_lens[min(int(len(tok_lens) * p), len(tok_lens) - 1)]

        # question-level bootstrap CI (reviewer item 9): AIME24 has only 30 questions,
        # so a point estimate at n=16 samples carries a large sampling error. Comparisons
        # against the paper's Table 3 must be made on the interval, not the point.
        import random as _rnd
        _rng = _rnd.Random(args.seed)

        def _boot_ci(pq):
            b = sorted(100 * st.mean([pq[_rng.randrange(len(pq))] for _ in range(len(pq))])
                       for _ in range(2000))
            return [round(b[int(0.025 * len(b))], 2), round(b[int(0.975 * len(b))], 2)]

        ci = _boot_ci(per_q_acc)
        ci_ne = _boot_ci(per_q_acc_ne)

        results[name] = {
            "n_questions": len(items),
            "pass@1": round(100 * st.mean(per_q_acc), 2),           # equal weight/question
            "pass@1_ci95": ci,
            # Primary accuracy: strict grader OR numeric equivalence.
            "pass@1_numeq": round(100 * st.mean(per_q_acc_ne), 2),
            "pass@1_numeq_ci95": ci_ne,
            "n_numeq_rescued": n_resc,
            "pass@1_std_over_questions": round(100 * st.pstdev(per_q_acc), 2),
            "exceed": round(100 * n_exceed / max(n_total, 1), 2),   # strict >4000
            "joint": round(100 * n_joint / max(n_total, 1), 2),     # correct (numeq) and <= 4000 tokens
            "tokens_mean": round(st.mean(tok_lens), 1),
            "tokens_p50": pct(0.50),
            "tokens_p95": pct(0.95),
            "n_samples": n_total,
        }
        r = results[name]
        print("%-10s n=%4d  pass@1=%6.2f%% [%.2f, %.2f]  numeq=%6.2f%% [%.2f, %.2f] (+%d)  "
              "Exceed=%6.2f%%  tok mean/p50/p95=%.0f/%d/%d"
              % (name, r["n_questions"], r["pass@1"], r["pass@1_ci95"][0],
                 r["pass@1_ci95"][1], r["pass@1_numeq"], r["pass@1_numeq_ci95"][0],
                 r["pass@1_numeq_ci95"][1], r["n_numeq_rescued"], r["exceed"],
                 r["tokens_mean"], r["tokens_p50"], r["tokens_p95"]))

    macro = {key: round(st.mean(r[key] for r in results.values()), 2)
             for key in ("pass@1_numeq", "exceed", "joint")}
    print("macro      Acc=%.2f%%  Exceed=%.2f%%  Joint=%.2f%%" % (macro["pass@1_numeq"], macro["exceed"], macro["joint"]))
    meta = {"ckpt": args.ckpt, "n_samples_per_q": args.n,
            "temperature": args.temperature, "top_p": args.top_p,
            "max_tokens": args.max_tokens, "length_limit": args.length_limit,
            "seed": args.seed, "macro": macro, "results": results}
    out = args.out or os.path.join(args.ckpt, "math_eval.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print("\n-> %s" % out)


if __name__ == "__main__":
    main()
