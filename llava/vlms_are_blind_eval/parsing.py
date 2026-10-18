import re
from dataclasses import dataclass
from typing import Any

YES_VALUES = {"yes", "y", "true"}
NO_VALUES = {"no", "n", "false"}


@dataclass(frozen=True)
class ParsedAnswer:
    value: Any
    raw: str
    ok: bool


def normalize_text(text: Any) -> str:
    return str(text).strip()


def normalize_yes_no(text: Any) -> str | None:
    cleaned = normalize_text(text).lower()
    bracketed = extract_bracketed(cleaned)
    if bracketed:
        cleaned = bracketed.lower()
    cleaned = cleaned.strip(" .,:;!?\"'")
    if cleaned in YES_VALUES:
        return "yes"
    if cleaned in NO_VALUES:
        return "no"
    match = re.search(r"\b(yes|no|true|false|y|n)\b", cleaned)
    if not match:
        return None
    return "yes" if match.group(1) in YES_VALUES else "no"


def extract_bracketed(text: str) -> str | None:
    match = re.search(r"\{([^{}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def parse_integer(text: Any) -> ParsedAnswer:
    raw = normalize_text(text)
    candidate = extract_bracketed(raw) or raw
    match = re.search(r"[-+]?\d+", candidate)
    if not match:
        return ParsedAnswer(None, raw, False)
    return ParsedAnswer(int(match.group(0)), raw, True)


def parse_letter(text: Any) -> ParsedAnswer:
    raw = normalize_text(text)
    candidate = extract_bracketed(raw) or raw
    match = re.search(r"[A-Za-z]", candidate)
    if not match:
        return ParsedAnswer(None, raw, False)
    return ParsedAnswer(match.group(0), raw, True)


def parse_yes_no(text: Any) -> ParsedAnswer:
    raw = normalize_text(text)
    value = normalize_yes_no(raw)
    return ParsedAnswer(value, raw, value is not None)


def parse_number_list(text: Any) -> ParsedAnswer:
    raw = normalize_text(text)
    # Extract numbers from ALL {…} brackets (e.g. "Rows={4} Columns={4}").
    # Fall back to searching the full string if no brackets are present.
    bracketed = re.findall(r"\{([^{}]+)\}", raw)
    candidate = " ".join(bracketed) if bracketed else raw
    nums = [int(x) for x in re.findall(r"[-+]?\d+", candidate)]
    return ParsedAnswer(tuple(nums), raw, bool(nums))


def infer_answer_kind(task: str, prompt: str, groundtruth: Any) -> str:
    gt = normalize_text(groundtruth)
    gt_lower = gt.lower()
    prompt_lower = normalize_text(prompt).lower()
    task_lower = normalize_text(task).lower()

    # Letter-task check must come before yes_no: single letters like "y" and "n"
    # are in YES_VALUES/NO_VALUES but are legitimate letter answers for these tasks.
    if "circled letter" in task_lower or "letter" in prompt_lower or "character" in prompt_lower:
        return "letter"
    if "row" in prompt_lower and "column" in prompt_lower:
        return "number_list"
    if gt_lower in YES_VALUES | NO_VALUES:
        return "yes_no"
    if re.fullmatch(r"[-+]?\d+", gt):
        return "integer"
    if len(gt) == 1 and gt.isalpha():
        return "letter"
    if re.search(r"\d", gt):
        return "number_list"
    return "text"


def parse_for_kind(text: Any, kind: str) -> ParsedAnswer:
    if kind == "yes_no":
        return parse_yes_no(text)
    if kind == "integer":
        return parse_integer(text)
    if kind == "letter":
        return parse_letter(text)
    if kind == "number_list":
        return parse_number_list(text)
    raw = normalize_text(text)
    return ParsedAnswer(raw.lower(), raw, bool(raw))


def is_correct(prediction: ParsedAnswer, groundtruth: Any, kind: str) -> bool:
    gold = parse_for_kind(groundtruth, kind)
    if not prediction.ok or not gold.ok:
        return False
    if kind == "letter":
        return str(prediction.value) == str(gold.value)
    return prediction.value == gold.value
