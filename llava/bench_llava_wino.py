import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model

WINOGROUND_DATASET_ID = "facebook/winoground"

# The 11 vision-encoder-family comparison models (checkpoint dir llava-v1.5-7b-lora-<name>).
VIT_FAMILY_NAMES = [
    'dinov1b', 'datacompclip', 'vitb', 'siglipb', 'siglip2b', 'beitb',
    'deitb', 'dinov2b', 'vit21b', 'maeb', 'clipb',
]

# The 26 fine-tuned-ViT candidate-pool models (checkpoint dir llava-v1.5-7b-lora-ft-<slug>);
# see slurm_scripts/vit_finetunes/gen_finetune_jobs.py.
FINETUNE_SLUGS = [
    "fairface_age", "rorshark_vit", "deepfake_v2", "man_woman_face",
    "facial_emotions", "deepfake_vs_real", "ai_vs_real", "watermark_vit",
    "fairface_gender", "chest_xray", "closed_eyes", "ul_base", "ul_interior",
    "ul_bedroom", "face_emotion", "blur_vs_clean", "human_action",
    "watermark_siglip", "age_siglip", "gender_mini", "flowers102",
    "oxfordpet", "cifar100", "fer2013", "cifar10", "stl10",
]


def paper_model_paths(ckpt_dir: str) -> List[str]:
    """The 37 checkpoints of fig:repr_align_vlm_failures, in evaluation order."""
    return ([os.path.join(ckpt_dir, f'llava-v1.5-7b-lora-{name}') for name in VIT_FAMILY_NAMES] +
            [os.path.join(ckpt_dir, f'llava-v1.5-7b-lora-ft-{slug}') for slug in FINETUNE_SLUGS])


def get_model_dtype(model) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float16


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token to use as pad token.")
    return tokenizer.pad_token_id


def build_prompt_text(caption: str) -> str:
    return (
        "Does the caption match the image?\n"
        f"Caption: {caption}\n"
        "Answer with only 'yes' or 'no'."
    )


def build_llava_prompt(model, prompt_text: str) -> str:
    """
    LLaVA v1.5 formatting:
      <image token> + newline + plain prompt
      wrapped in the llava_v1 conversation template
    """
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    else:
        image_token = DEFAULT_IMAGE_TOKEN

    qs = f"{image_token}\n{prompt_text}"

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def resolve_token_ids(tokenizer, variants: Sequence[str]) -> torch.Tensor:
    """
    Resolve a set of token string variants to unique single-token ids.
    Variants that split into multiple tokens are ignored.
    """
    ids = []
    seen = set()

    for variant in variants:
        tok_ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(tok_ids) != 1:
            continue
        tid = tok_ids[0]
        if tid not in seen:
            seen.add(tid)
            ids.append(tid)

    if not ids:
        raise ValueError("Could not resolve any single-token yes/no variants for this tokenizer.")

    return torch.tensor(ids, dtype=torch.long)


def evaluate_scores(scores: torch.Tensor) -> Tuple[bool, bool, bool]:
    """
    scores is a 2x2 matrix:
      rows    = images [image_0, image_1]
      columns = captions [caption_0, caption_1]

    Winoground:
      text_correct  : each caption prefers its matching image
      image_correct : each image prefers its matching caption
      group_correct : both
    """
    if scores.shape != (2, 2):
        raise ValueError(f"Expected scores shape (2, 2), got {tuple(scores.shape)}")

    s00, s01 = scores[0, 0], scores[0, 1]
    s10, s11 = scores[1, 0], scores[1, 1]

    text_correct = bool((s00 > s10) and (s11 > s01))
    image_correct = bool((s00 > s01) and (s11 > s10))
    group_correct = bool(text_correct and image_correct)

    return text_correct, image_correct, group_correct


@torch.inference_mode()
def score_items(
    model,
    tokenizer,
    image_processor,
    items: List[Dict],
    yes_token_ids: torch.Tensor,
    no_token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Score a batch of (image, caption) items using next-token logit mass:
      score = logsumexp(yes logits) - logsumexp(no logits)
    """
    device = model.device
    dtype = get_model_dtype(model)

    prompts = []
    images = []

    for item in items:
        prompt_text = build_prompt_text(item["caption"])
        prompts.append(build_llava_prompt(model, prompt_text))
        images.append(item["image"])

    # Tokenize each prompt separately because captions differ in length
    prompt_ids = [
        tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).squeeze(0)
        for prompt in prompts
    ]

    pad_id = ensure_pad_token(tokenizer)
    input_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=pad_id).to(device)
    attention_mask = input_ids.ne(pad_id).long()

    image_sizes = [img.size for img in images]
    images_tensor = process_images(images, image_processor, model.config).to(device=device, dtype=dtype)

    (
        _,
        position_ids_mm,
        attention_mask_mm,
        _,
        inputs_embeds_mm,
        _,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=images_tensor,
        image_sizes=image_sizes,
    )

    outputs = model(
        input_ids=None,
        inputs_embeds=inputs_embeds_mm,
        position_ids=position_ids_mm,
        attention_mask=attention_mask_mm,
        use_cache=False,
    )

    last_indices = attention_mask_mm.sum(dim=1) - 1
    batch_indices = torch.arange(input_ids.shape[0], device=device)
    last_logits = outputs.logits[batch_indices, last_indices]

    yes_score = torch.logsumexp(last_logits[:, yes_token_ids.to(device)], dim=-1)
    no_score = torch.logsumexp(last_logits[:, no_token_ids.to(device)], dim=-1)

    return (yes_score - no_score).detach().cpu()


def build_batch_from_examples(examples: Sequence[dict]) -> list[dict]:
    """
    For each Winoground example, produce 4 pairs in this order:
      (image_0, caption_0)
      (image_0, caption_1)
      (image_1, caption_0)
      (image_1, caption_1)
    """
    items = []

    for ex in examples:
        img0 = ex["image_0"].convert("RGB")
        img1 = ex["image_1"].convert("RGB")
        cap0 = ex["caption_0"]
        cap1 = ex["caption_1"]

        items.extend(
            [
                {"id": ex["id"], "image": img0, "caption": cap0},
                {"id": ex["id"], "image": img0, "caption": cap1},
                {"id": ex["id"], "image": img1, "caption": cap0},
                {"id": ex["id"], "image": img1, "caption": cap1},
            ]
        )

    return items


def score_examples(
    model,
    tokenizer,
    image_processor,
    examples: Sequence[dict],
    yes_token_ids: torch.Tensor,
    no_token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Returns tensor of shape [num_examples, 2, 2].
    """
    items = build_batch_from_examples(examples)
    pair_scores = score_items(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        items=items,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
    )
    return pair_scores.view(len(examples), 2, 2)


def evaluate_winoground_llava(
    model,
    tokenizer,
    image_processor,
    split: str = "test",
    batch_size_examples: int = 8,
    save_path: str | None = None,
) -> dict:
    """
    Evaluate a loaded LLaVA model on Winoground using plain yes/no logit scoring.

    Assumptions:
      - model, tokenizer, image_processor are already loaded
      - model is a LLaVA v1.5-style model
    """
    tokenizer.padding_side = "right"
    ensure_pad_token(tokenizer)

    yes_variants = [
        "yes", "Yes", "YES", "y", "Y",
    ]
    no_variants = [
        "no", "No", "NO", "n", "N",
    ]
    yes_token_ids = resolve_token_ids(tokenizer, yes_variants)
    no_token_ids = resolve_token_ids(tokenizer, no_variants)

    dset = load_dataset(WINOGROUND_DATASET_ID, split=split, token=os.environ.get("HF_TOKEN"))
    print(f"Loaded {len(dset)} Winoground examples from split={split!r}")

    results = []
    pending = []

    for ex in tqdm(dset, desc="Evaluating Winoground"):
        pending.append(ex)

        if len(pending) < batch_size_examples:
            continue

        scores_batch = score_examples(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            examples=pending,
            yes_token_ids=yes_token_ids,
            no_token_ids=no_token_ids,
        )

        for ex_i, scores in zip(pending, scores_batch):
            text_correct, image_correct, group_correct = evaluate_scores(scores)

            row = {
                "id": ex_i["id"],
                "scores": scores.tolist(),
                "text_correct": text_correct,
                "image_correct": image_correct,
                "group_correct": group_correct,
            }
            results.append(row)

        pending = []

    if pending:
        scores_batch = score_examples(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            examples=pending,
            yes_token_ids=yes_token_ids,
            no_token_ids=no_token_ids,
        )

        for ex_i, scores in zip(pending, scores_batch):
            text_correct, image_correct, group_correct = evaluate_scores(scores)

            row = {
                "id": ex_i["id"],
                "scores": scores.tolist(),
                "text_correct": text_correct,
                "image_correct": image_correct,
                "group_correct": group_correct,
            }
            results.append(row)

    total = len(results)
    text_total = sum(r["text_correct"] for r in results)
    image_total = sum(r["image_correct"] for r in results)
    group_total = sum(r["group_correct"] for r in results)

    summary = {
        "total": total,
        "text_score": text_total / total if total else 0.0,
        "image_score": image_total / total if total else 0.0,
        "group_score": group_total / total if total else 0.0,
        "text_total": text_total,
        "image_total": image_total,
        "group_total": group_total,
    }

    print("\n=== Winoground Scores ===")
    print(f"Text score : {summary['text_score']:.4f} ({text_total}/{total})")
    print(f"Image score: {summary['image_score']:.4f} ({image_total}/{total})")
    print(f"Group score: {summary['group_score']:.4f} ({group_total}/{total})")

    if save_path is not None:
        save_path = str(save_path)
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Saved results to {out_path}")

    return {
        "summary": summary,
        "results": results,
    }


def save_path_for(model_path: str, output_dir: str = ".") -> str:
    model_suffix = model_path.rstrip("/").split("/")[-1].split('lora-')[-1].replace('-', '_')
    return os.path.join(output_dir, f"winoground_llava_{model_suffix}_res.jsonl")


def run_single(model_path: str, model_base: str, save_path: str, batch_size_examples: int = 8) -> None:
    """Load one model, run the Winoground eval, save results"""
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                           device_map='cuda')

    model.eval()
    model.to('cuda')
    model.half()

    out = evaluate_winoground_llava(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        split="test",
        batch_size_examples=batch_size_examples,
        save_path=save_path,
    )

    print(f'Summary for {model_path}', flush=True)
    print(out['summary'], flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_paths', nargs='+', default=None,
                        help='Checkpoints to evaluate. Default: the 37 paper models under --ckpt_dir.')
    parser.add_argument('--ckpt_dir', default=os.environ.get('CKPT_DIR'),
                        help='Directory with the llava-v1.5-7b-lora-* checkpoints (default: $CKPT_DIR). '
                             'Used only when --model_paths is not given.')
    parser.add_argument('--model_base', default='lmsys/vicuna-7b-v1.5',
                        help='Base LLM the LoRA checkpoints were trained on.')
    parser.add_argument('--output_dir', default='.', help='Where winoground_llava_*_res.jsonl files are written.')
    parser.add_argument('--batch_size_examples', type=int, default=8,
                        help='Winoground examples per forward pass (4 image-caption pairs each).')
    parser.add_argument('--single', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.model_paths is None:
        if not args.ckpt_dir:
            parser.error('pass --model_paths or --ckpt_dir (or set CKPT_DIR)')
        args.model_paths = paper_model_paths(args.ckpt_dir)
    return args


def main(args):
    for model_path in args.model_paths:
        save_path = save_path_for(model_path, args.output_dir)

        if Path(save_path).exists():
            print(f"=== Skipping {model_path}: {save_path} already exists ===", flush=True)
            continue

        print(f"\n\n=== Evaluating model at {model_path} ===\n", flush=True)

        # Run in a fresh subprocess per model, for crash resistance
        result = subprocess.run(
            [sys.executable, __file__, "--single", "--model_paths", model_path,
             "--model_base", args.model_base, "--output_dir", args.output_dir,
             "--batch_size_examples", str(args.batch_size_examples)],
            env=os.environ,
        )
        if result.returncode != 0:
            print(f"ERROR evaluating {model_path}: subprocess exited with code {result.returncode}", flush=True)


if __name__ == '__main__':
    args = parse_args()
    if args.single:
        model_path = args.model_paths[0]
        run_single(model_path, args.model_base, save_path_for(model_path, args.output_dir), args.batch_size_examples)
    else:
        main(args)
