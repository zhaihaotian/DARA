#!/usr/bin/env python3
"""Numeric-equivalence fallback for math evaluation.

The Hendrycks MATH grader compares normalized strings, so it misses answers that are the
same number written differently (e.g. '025' vs 25, '4.5e33' vs 4.5\\times10^{33}, '0.10' vs 0.1).
`numeq` is only consulted when the strict grader returns 0, so it can turn a miss into a hit
but never the reverse. It accepts exact numeric equality only (no tolerance by default) and
does not try to compare fractions, radicals or other expressions.
"""
import re

# unit wrappers such as \text{...} and \mathrm{...}
_UNIT = re.compile(r"\\(?:text|mathrm|mbox|rm|textrm)\s*\{[^{}]*\}")
_STRIP = ["\\!", "\\,", "\\ ", "\\;", "\\:", "~", "$", ",", "%",
          "^\\circ", "\\circ", "\\%", "\\left", "\\right"]
# a \times 10^{n} / a \cdot 10^n  ->  aen
_SCI = re.compile(r"\\(?:times|cdot|ast)\s*10\s*\^\s*\{?\s*(-?\+?\d+)\s*\}?")
_BARE10 = re.compile(r"^10\s*\^\s*\{?\s*(-?\+?\d+)\s*\}?$")
_NUM = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")

BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

# Outer LaTeX math delimiters around a ground truth (OlympiadBench stores e.g. '$2^{1009}$').
# They are kept when the content is empty or contains the same delimiter (e.g. '$a$ or $b$').
_DELIM = (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"))


def unwrap_math(g):
    """Strip matching outer math delimiters from a ground truth; return it unchanged when unsure."""
    t = str(g).strip()
    for _ in range(3):                       # at most three nested layers
        s = t
        for a, b in _DELIM:
            if len(t) > len(a) + len(b) and t.startswith(a) and t.endswith(b):
                inner = t[len(a):len(t) - len(b)].strip()
                if inner and a not in inner and b not in inner:
                    t = inner
        if t == s:
            break
    return t


def to_num(s):
    """Parse an answer string as a float; return None whenever it is not a plain number."""
    if s is None:
        return None
    t = _UNIT.sub("", str(s).strip())
    for a in _STRIP:
        t = t.replace(a, "")
    t = _SCI.sub(lambda m: "e" + m.group(1).lstrip("+"), t)
    t = re.sub(r"\s+", "", t)
    t = t.rstrip(".")
    if _BARE10.match(t):                      # 10^{11} -> 1e11
        t = "1e" + _BARE10.match(t).group(1).lstrip("+")
    if not _NUM.match(t):                     # fractions, radicals, expressions, multiple answers
        return None
    try:
        v = float(t)
    except Exception:
        return None
    return v if v == v and abs(v) != float("inf") else None   # exclude nan / inf


def numeq(resp_text, gt, rtol=0.0):
    """True when the last \\boxed{} answer in the response equals the ground truth numerically."""
    m = BOXED.findall(resp_text or "")
    if not m:
        return False
    a, g = to_num(m[-1]), to_num(gt)
    if a is None or g is None:
        return False
    if rtol <= 0:
        return a == g
    d = max(abs(a), abs(g))
    return abs(a - g) <= rtol * d if d else abs(a - g) < 1e-12
