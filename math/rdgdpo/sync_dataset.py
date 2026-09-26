"""Text dataset adapter for the paper's synchronous vLLM rollout."""

import warnings

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


def sync_text_getitem(dataset, item):
    row_dict: dict = dataset.dataframe[item]
    multimodal_keys = (getattr(dataset, "image_key", "images"),
                       getattr(dataset, "video_key", "videos"), "multi_modal_data")
    has_media = any(row_dict.get(key) is not None and len(row_dict[key]) > 0
                    for key in multimodal_keys)
    if dataset.processor is not None or has_media:
        raise NotImplementedError("The RD-GDPO paper runtime supports text-only datasets")
    messages = dataset._build_messages(row_dict)

    apply_kwargs = dict(**dataset.apply_chat_template_kwargs)
    if dataset.tool_schemas is not None:
        apply_kwargs["tools"] = dataset.tool_schemas
    raw_prompt = dataset.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
    )
    model_inputs = dataset.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids, attention_mask = verl_F.postprocess_data(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_length=dataset.max_prompt_length,
        pad_token_id=dataset.tokenizer.pad_token_id,
        left_pad=True,
        truncation=dataset.truncation,
    )
    row_dict["input_ids"] = input_ids[0]
    row_dict["attention_mask"] = attention_mask[0]
    row_dict["position_ids"] = compute_position_id_with_mask(attention_mask)[0]

    raw_prompt_ids = dataset.tokenizer.encode(raw_prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > dataset.max_prompt_length:
        if dataset.truncation == "left":
            raw_prompt_ids = raw_prompt_ids[-dataset.max_prompt_length :]
        elif dataset.truncation == "right":
            raw_prompt_ids = raw_prompt_ids[: dataset.max_prompt_length]
        elif dataset.truncation == "middle":
            left_half = dataset.max_prompt_length // 2
            right_half = dataset.max_prompt_length - left_half
            raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
        else:
            raise RuntimeError(
                f"Prompt length {len(raw_prompt_ids)} is longer than {dataset.max_prompt_length}."
            )
    row_dict["raw_prompt_ids"] = raw_prompt_ids
    if dataset.return_raw_chat:
        row_dict["raw_prompt"] = messages
    if dataset.return_full_prompt:
        row_dict["full_prompts"] = raw_prompt

    if "extra_info" not in row_dict or row_dict["extra_info"] is None:
        row_dict["extra_info"] = {}
    index = row_dict.get("extra_info", {}).get("index", 0)
    tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
    interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
    need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", dataset.need_tools_kwargs)
    if need_tools_kwargs and not tools_kwargs:
        warnings.warn(f"tools_kwargs is empty for dataset index {index}", stacklevel=2)
    row_dict["index"] = index
    row_dict["tools_kwargs"] = tools_kwargs
    row_dict["interaction_kwargs"] = interaction_kwargs
    return row_dict
