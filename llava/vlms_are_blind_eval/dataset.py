from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

DATASET_ID = "XAI/vlmsareblind"
DEFAULT_SPLIT = "valid"


@dataclass(frozen=True)
class BlindExample:
    index: int
    task: str
    image: object
    prompt: str
    groundtruth: str
    metadata: str


def load_blind_dataset(dataset_id: str = DATASET_ID, split: str = DEFAULT_SPLIT):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face `datasets` package is required. Install it in the "
            "environment used to run this evaluator, e.g. `pip install datasets`."
        ) from exc
    return load_dataset(dataset_id, split=split)


def iter_examples(dataset, task: str | None = None, limit: int | None = None) -> Iterable[BlindExample]:
    seen = 0
    for idx, row in enumerate(dataset):
        if task is not None and row["task"] != task:
            continue
        yield BlindExample(
            index=idx,
            task=row["task"],
            image=row["image"],
            prompt=row["prompt"],
            groundtruth=str(row["groundtruth"]),
            metadata=row.get("metadata", ""),
        )
        seen += 1
        if limit is not None and seen >= limit:
            break
