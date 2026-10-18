from __future__ import annotations

from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def get_model_dtype(model) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float16


def ensure_pad_token(tokenizer) -> int:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token to use as pad token.")
    return tokenizer.pad_token_id


def build_llava_prompt(model, prompt_text: str, conv_mode: str = "llava_v1") -> str:
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    else:
        image_token = DEFAULT_IMAGE_TOKEN

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], f"{image_token}\n{prompt_text}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def resolve_single_token_ids(tokenizer, variants: Sequence[str]) -> torch.Tensor:
    ids = []
    seen = set()
    for variant in variants:
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        token_id = token_ids[0]
        if token_id not in seen:
            seen.add(token_id)
            ids.append(token_id)
    if not ids:
        raise ValueError(f"No variants resolved to a single token: {variants}")
    return torch.tensor(ids, dtype=torch.long)


def load_llava(model_path: str, model_base: str | None, device: str = "cuda"):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        device_map=device,
    )
    model.eval()
    if device == "cuda":
        model.to("cuda")
        model.half()
    return tokenizer, model, image_processor, context_len


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    image_processor,
    image,
    prompt: str,
    conv_mode: str,
    temperature: float,
    top_p: float | None,
    num_beams: int,
    max_new_tokens: int,
) -> str:
    image = image.convert("RGB")
    full_prompt = build_llava_prompt(model, prompt, conv_mode=conv_mode)
    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).to(device=model.device, dtype=get_model_dtype(model))

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    if output_ids.shape[1] > input_ids.shape[1]:
        output_ids = output_ids[:, input_ids.shape[1] :]
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


@torch.inference_mode()
def score_yes_no_batch(
    model,
    tokenizer,
    image_processor,
    examples,
    conv_mode: str,
    yes_token_ids: torch.Tensor,
    no_token_ids: torch.Tensor,
) -> list[dict]:
    tokenizer.padding_side = "right"
    pad_id = ensure_pad_token(tokenizer)
    prompts = [build_llava_prompt(model, ex.prompt, conv_mode=conv_mode) for ex in examples]
    images = [ex.image.convert("RGB") for ex in examples]
    input_id_list = [
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").squeeze(0)
        for prompt in prompts
    ]
    input_ids = pad_sequence(input_id_list, batch_first=True, padding_value=pad_id).to(model.device)
    attention_mask = input_ids.ne(pad_id).long()
    image_sizes = [image.size for image in images]
    image_tensor = process_images(images, image_processor, model.config).to(
        device=model.device,
        dtype=get_model_dtype(model),
    )

    _, position_ids, attention_mask_mm, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=image_tensor,
        image_sizes=image_sizes,
    )
    outputs = model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=attention_mask_mm,
        use_cache=False,
    )
    last_indices = attention_mask_mm.sum(dim=1) - 1
    batch_indices = torch.arange(input_ids.shape[0], device=model.device)
    logits = outputs.logits[batch_indices, last_indices]
    yes_scores = torch.logsumexp(logits[:, yes_token_ids.to(model.device)], dim=-1)
    no_scores = torch.logsumexp(logits[:, no_token_ids.to(model.device)], dim=-1)
    margins = yes_scores - no_scores

    rows = []
    for yes_score, no_score, margin in zip(yes_scores, no_scores, margins):
        rows.append(
            {
                "prediction": "yes" if margin.item() > 0 else "no",
                "yes_logit_score": float(yes_score.item()),
                "no_logit_score": float(no_score.item()),
                "yes_no_margin": float(margin.item()),
            }
        )
    return rows
