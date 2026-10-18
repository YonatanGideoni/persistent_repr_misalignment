import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    CLIPImageProcessor,
    CLIPModel,
    CLIPVisionModel,
    SiglipVisionModel,
    ViTMAEModel,
)

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "platonic-rep"))

from alignment_eval import (  # noqa: E402
    _load_wit1024_images_and_texts,
    compute_alignment_score,
    compute_or_load_vicuna_feats,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = REPO_ROOT / "alignment_scores"

# 11 base backbones, loaded as plain vision models
BASE_MODELS = [
    "facebook/dino-vitb16",
    "facebook/deit-base-distilled-patch16-224",
    "facebook/vit-mae-base",
    "microsoft/beit-base-patch16-224",
    "facebook/dinov2-base",
    "openai/clip-vit-base-patch16",
    "flavour/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    "google/siglip-base-patch16-224",
    "google/siglip2-base-patch16-224",
    "google/vit-base-patch16-224",
    "google/vit-base-patch16-224-in21k",
]


def load_finetune_list():
    """[(hf_id, family)] for the 26 fine-tuned backbones, from the LLaVA job generator."""
    path = REPO_ROOT.parents[1] / "llava" / "slurm_scripts" / "vit_finetunes" / "gen_finetune_jobs.py"
    spec = importlib.util.spec_from_file_location("gen_finetune_jobs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [(m["id"], m["family"]) for m in mod.MODELS]


# ---------------------------------------------------------------------------
# Model loading -- plain loaders for the base backbones and per-family loaders for the
# fine-tuned classifiers.
# ---------------------------------------------------------------------------

def load_pool_model(model_name):
    if "clip" in model_name and "openai" in model_name:
        model = CLIPVisionModel.from_pretrained(model_name)
    elif "mae" in model_name:
        model = ViTMAEModel.from_pretrained(model_name)
        model.config.mask_ratio = 0.0  # disable MAE's random patch masking
    elif "siglip" in model_name:
        model = SiglipVisionModel.from_pretrained(model_name)
    elif "clip" in model_name.lower() and "datacomp" in model_name.lower():
        model = CLIPVisionModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model.to(DEVICE).eval(), processor


class _VisionWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.vision_model = transformer

    def forward(self, pixel_values, output_hidden_states=False, **kwargs):
        return self.vision_model(pixel_values, output_hidden_states=output_hidden_states, **kwargs)


def load_vit_candidate(model_id):
    processor = AutoImageProcessor.from_pretrained(model_id)
    try:
        cfg = AutoConfig.from_pretrained(model_id)
        arch = (cfg.architectures or [""])[0]
    except Exception:
        arch = ""
    if "ForImageClassification" in arch:
        clf = AutoModelForImageClassification.from_pretrained(model_id, ignore_mismatched_sizes=True)
        if hasattr(clf, "vit"):
            return clf.vit.eval(), processor
        if hasattr(clf, "model"):
            return clf.model.eval(), processor
        return clf.eval(), processor
    return AutoModel.from_pretrained(model_id, ignore_mismatched_sizes=True).eval(), processor


def load_siglip_candidate(model_id):
    processor = AutoImageProcessor.from_pretrained(model_id)
    try:
        return SiglipVisionModel.from_pretrained(model_id).eval(), processor
    except Exception:
        pass
    clf = AutoModelForImageClassification.from_pretrained(model_id, ignore_mismatched_sizes=True)
    for attr_path in ["siglip.vision_model", "model.vision_model", "vision_model"]:
        try:
            part = clf
            for a in attr_path.split("."):
                part = getattr(part, a)
            return _VisionWrapper(part).eval(), processor
        except AttributeError:
            continue
    raise RuntimeError(f"cannot extract SigLIP vision backbone from '{model_id}'")


def load_clip_candidate(model_id):
    try:
        processor = CLIPImageProcessor.from_pretrained(model_id)
    except Exception:
        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
        except Exception:
            processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    try:
        return CLIPVisionModel.from_pretrained(model_id, ignore_mismatched_sizes=True).eval(), processor
    except Exception:
        pass
    clip = CLIPModel.from_pretrained(model_id, ignore_mismatched_sizes=True)
    return _VisionWrapper(clip.vision_model).eval(), processor


CANDIDATE_LOADERS = {"vit": load_vit_candidate, "siglip": load_siglip_candidate, "clip": load_clip_candidate}


def safe_name(model_id):
    return model_id.replace("/", "__")


def main(shard_index, num_shards, out_dir):
    tasks = [(m, None) for m in BASE_MODELS] + load_finetune_list()
    tasks = tasks[shard_index::num_shards]
    print(f"[shard {shard_index}/{num_shards}] {len(tasks)} tasks: {[t[0] for t in tasks]}")

    llm_feats = compute_or_load_vicuna_feats()
    images, _ = _load_wit1024_images_and_texts()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_id, family in tasks:
        out_path = out_dir / f"{safe_name(model_id)}.json"
        if out_path.exists():
            print(f"skip {model_id} (already computed)")
            continue
        print(f"computing alignment for {model_id} (family={family})")
        try:
            if family is None:
                model, processor = load_pool_model(model_id)
            else:
                model, processor = CANDIDATE_LOADERS[family](model_id)
            model = model.to(DEVICE).eval()
            result = compute_alignment_score(model, processor, llm_feats, images=images)
            result["model_id"] = model_id
            result["family"] = family
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(
                f"  -> score={result['score']:.4f} "
                f"llm_layer={result['best_llm_layer']} lvm_layer={result['best_lvm_layer']}"
            )
            del model
            torch.cuda.empty_cache()
        except Exception:
            print(f"  FAILED on {model_id}:")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    main(args.shard_index, args.num_shards, args.out_dir)
