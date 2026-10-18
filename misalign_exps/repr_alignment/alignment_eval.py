import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import trange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "platonic-rep"))
from metrics import AlignmentMetrics  # noqa: E402
from models import load_llm, load_tokenizer  # noqa: E402

LLM_MODEL_NAME = "lmsys/vicuna-7b-v1.5"
DATASET = "minhuh/prh"
SUBSET = "wit_1024"
CAPTION_IDX = 0
METRIC = "mutual_knn"
TOPK = 10
LLM_BATCH = 16
LVM_BATCH = 32

DEFAULT_CACHE_PATH = os.environ.get(
    "REPR_ALIGN_CACHE", str(Path(__file__).resolve().parent / "cache" / "vicuna_feats.pt")
)


def _load_wit1024_images_and_texts():
    from datasets import load_dataset

    dataset = load_dataset(DATASET, revision=SUBSET, split="train")
    texts = [str(x["text"][CAPTION_IDX]) for x in dataset]
    images = [x["image"] for x in dataset]
    return images, texts


def compute_or_load_vicuna_feats(cache_path: str = DEFAULT_CACHE_PATH, device: str = None) -> torch.Tensor:
    """Returns llm_feats of shape [N, L_llm, D_llm]. Cached to disk after
    the first call (per-process and across processes/runs)."""
    if os.path.exists(cache_path):
        print(f"Loading cached Vicuna features from {cache_path}")
        return torch.load(cache_path)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    images, texts = _load_wit1024_images_and_texts()

    print(f"Loading LLM: {LLM_MODEL_NAME}")
    language_model = load_llm(LLM_MODEL_NAME, qlora=False)
    llm_device = next(language_model.parameters()).device
    tokenizer = load_tokenizer(LLM_MODEL_NAME)

    tokens = tokenizer(texts, padding="longest", return_tensors="pt")

    llm_feats = []
    for i in trange(0, len(texts), LLM_BATCH, desc="LLM features"):
        token_inputs = {k: v[i:i + LLM_BATCH].to(llm_device).long() for (k, v) in tokens.items()}
        with torch.no_grad():
            llm_output = language_model(
                input_ids=token_inputs["input_ids"],
                attention_mask=token_inputs["attention_mask"],
            )
        feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3).cpu()
        mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1).cpu()
        feats = (feats * mask).sum(2) / mask.sum(2)
        llm_feats.append(feats)

    llm_feats = torch.cat(llm_feats)  # [N, L_llm, D_llm]
    print(f"LLM features: {tuple(llm_feats.shape)}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(llm_feats, cache_path)

    del language_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return llm_feats


@torch.no_grad()
def extract_vit_features(model, processor, images, device: str = None, batch_size: int = LVM_BATCH) -> torch.Tensor:
    """Mean-pools every transformer-block hidden state (mean over the token dimension, not the CLS token)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    lvm_feats = []
    for i in trange(0, len(images), batch_size, desc="ViT features"):
        batch_images = images[i:min(i + batch_size, len(images))]
        pixel_values = processor(images=batch_images, return_tensors="pt").pixel_values.to(device)
        outputs = model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
        feats = torch.stack([layer.mean(dim=1) for layer in hidden_states]).permute(1, 0, 2)
        lvm_feats.append(feats.cpu())

    return torch.cat(lvm_feats)  # [N, L_lvm, D_lvm]


def compute_alignment_score(model, processor, llm_feats: torch.Tensor, images=None) -> dict:
    """Searches all (LLM layer, ViT layer) pairs for the best mutual_knn score. Returns
    {"score": float, "best_llm_layer": int, "best_lvm_layer": int}."""
    if images is None:
        images, _ = _load_wit1024_images_and_texts()

    lvm_feats = extract_vit_features(model, processor, images)

    best_score, best_llm_layer, best_lvm_layer = -1.0, -1, -1
    n_llm_layers, n_lvm_layers = llm_feats.shape[1], lvm_feats.shape[1]

    for li in trange(n_llm_layers, desc="LLM layers"):
        for vi in range(n_lvm_layers):
            feats_a = F.normalize(llm_feats[:, li, :].float(), dim=-1)
            feats_b = F.normalize(lvm_feats[:, vi, :].float(), dim=-1)
            score = AlignmentMetrics.measure(METRIC, feats_a, feats_b, topk=TOPK)
            if score > best_score:
                best_score, best_llm_layer, best_lvm_layer = score, li, vi

    return {"score": best_score, "best_llm_layer": best_llm_layer, "best_lvm_layer": best_lvm_layer}


if __name__ == "__main__":
    feats = compute_or_load_vicuna_feats()
    print("Vicuna feats ready:", tuple(feats.shape))
