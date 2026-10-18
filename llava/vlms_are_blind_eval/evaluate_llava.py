from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from vlms_are_blind_eval.dataset import DEFAULT_SPLIT, DATASET_ID, iter_examples, load_blind_dataset
from vlms_are_blind_eval.llava_utils import (
    generate_answer,
    load_llava,
    resolve_single_token_ids,
    score_yes_no_batch,
)
from vlms_are_blind_eval.parsing import infer_answer_kind, is_correct, parse_for_kind


def flush_yes_no_batch(
    model,
    tokenizer,
    image_processor,
    examples,
    conv_mode,
    yes_token_ids,
    no_token_ids,
):
    scored = score_yes_no_batch(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        examples=examples,
        conv_mode=conv_mode,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
    )
    rows = []
    for ex, score_row in zip(examples, scored):
        parsed = parse_for_kind(score_row["prediction"], "yes_no")
        rows.append(
            {
                "index": ex.index,
                "task": ex.task,
                "prompt": ex.prompt,
                "groundtruth": ex.groundtruth,
                "answer_kind": "yes_no",
                "scoring": "yes_no_logits",
                "raw_prediction": score_row["prediction"],
                "parsed_prediction": parsed.value,
                "correct": is_correct(parsed, ex.groundtruth, "yes_no"),
                "metadata": ex.metadata,
                **score_row,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    totals = Counter()
    correct = Counter()
    by_kind = Counter()
    by_kind_correct = Counter()
    for row in rows:
        key = row["task"]
        totals[key] += 1
        correct[key] += int(row["correct"])
        kind = row["answer_kind"]
        by_kind[kind] += 1
        by_kind_correct[kind] += int(row["correct"])
    overall_total = len(rows)
    overall_correct = sum(1 for row in rows if row["correct"])
    per_task = {
        task: {
            "total": totals[task],
            "correct": correct[task],
            "accuracy": correct[task] / totals[task] if totals[task] else 0.0,
        }
        for task in sorted(totals)
    }
    per_kind = {
        kind: {
            "total": by_kind[kind],
            "correct": by_kind_correct[kind],
            "accuracy": by_kind_correct[kind] / by_kind[kind] if by_kind[kind] else 0.0,
        }
        for kind in sorted(by_kind)
    }
    return {
        "total": overall_total,
        "correct": overall_correct,
        "accuracy": overall_correct / overall_total if overall_total else 0.0,
        "per_task": per_task,
        "per_answer_kind": per_kind,
    }


def evaluate(args) -> dict:
    dataset = load_blind_dataset(args.dataset_id, args.split)
    tokenizer, model, image_processor, _ = load_llava(args.model_path, args.model_base, args.device)
    yes_token_ids = resolve_single_token_ids(tokenizer, ["yes", "Yes", "YES", "y", "Y"])
    no_token_ids = resolve_single_token_ids(tokenizer, ["no", "No", "NO", "n", "N"])

    rows = []
    yes_no_pending = []
    iterator = iter_examples(dataset, task=args.task, limit=args.limit)

    for ex in tqdm(iterator, desc="Evaluating VLMs Are Blind"):
        kind = infer_answer_kind(ex.task, ex.prompt, ex.groundtruth)
        if kind == "yes_no":
            yes_no_pending.append(ex)
            if len(yes_no_pending) >= args.batch_size:
                rows.extend(
                    flush_yes_no_batch(
                        model,
                        tokenizer,
                        image_processor,
                        yes_no_pending,
                        args.conv_mode,
                        yes_token_ids,
                        no_token_ids,
                    )
                )
                yes_no_pending = []
            continue

        if yes_no_pending:
            rows.extend(
                flush_yes_no_batch(
                    model,
                    tokenizer,
                    image_processor,
                    yes_no_pending,
                    args.conv_mode,
                    yes_token_ids,
                    no_token_ids,
                )
            )
            yes_no_pending = []

        raw_prediction = generate_answer(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image=ex.image,
            prompt=ex.prompt,
            conv_mode=args.conv_mode,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
        parsed = parse_for_kind(raw_prediction, kind)
        rows.append(
            {
                "index": ex.index,
                "task": ex.task,
                "prompt": ex.prompt,
                "groundtruth": ex.groundtruth,
                "answer_kind": kind,
                "scoring": "generation_parse",
                "raw_prediction": raw_prediction,
                "parsed_prediction": parsed.value,
                "correct": is_correct(parsed, ex.groundtruth, kind),
                "metadata": ex.metadata,
            }
        )

    if yes_no_pending:
        rows.extend(
            flush_yes_no_batch(
                model,
                tokenizer,
                image_processor,
                yes_no_pending,
                args.conv_mode,
                yes_token_ids,
                no_token_ids,
            )
        )

    summary = summarize(rows)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    del model
    torch.cuda.empty_cache()
    return {"summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--task", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="vlms_are_blind_results.jsonl")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
