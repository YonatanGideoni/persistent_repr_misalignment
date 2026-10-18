import os

from .clip_encoder import (
    CLIPVisionTower, CLIPVisionTowerS2, ViTVisionTower, MAEVisionTower,
    DINOv2VisionTower, GenericVisionTower, SiglipVisionTower, CLIPAltVisionTower,
)


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)

    # Explicit family override — set via --mm_vision_tower_type in training scripts.
    # Routes fine-tuned models to the same tower class as their parent architecture.
    # CLIPVisionTower already handles missing preprocessors with a fallback.
    tower_type = getattr(vision_tower_cfg, 'mm_vision_tower_type', 'auto')
    if tower_type == 'vit':
        return ViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if tower_type == 'siglip':
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if tower_type == 'clip':
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # Auto-detection by model name (existing / non-finetuned models)
    if (is_absolute_path_exists or vision_tower.startswith("openai") or
        vision_tower.startswith("laion") or
        "ShareGPT4V" in vision_tower):
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if vision_tower.startswith("google") and 'vit' in vision_tower.lower():
        return ViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if 'siglip' in vision_tower.lower():
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if 'mae' in vision_tower.lower():
        return MAEVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if 'dinov2' in vision_tower.lower():
        return DINOv2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if 'clip' in vision_tower.lower() and 'datacomp' in vision_tower.lower():
        return CLIPAltVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    return GenericVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
