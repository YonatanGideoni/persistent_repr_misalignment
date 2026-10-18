from collections import defaultdict

import numpy as np
import open_clip
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from coco_utils import load_datasets
from constants import DEF_COCO_DIR, device

N_IMAGES = 1000


def l2_normalize(x: np.ndarray, eps=1e-15):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def gather_images_and_captions(train_ds, n_gather=N_IMAGES):
    images = {}
    captions = defaultdict(list)
    seen_img_ids = set()

    i = 0
    while True:
        example = train_ds[i]
        imgid = example["imgid"]

        i += 1

        if len(images) == n_gather and imgid not in seen_img_ids:
            break

        caption = example["sentences"]["raw"].strip()
        captions[imgid].append(caption)

        if imgid in seen_img_ids:
            continue
        seen_img_ids.add(imgid)

        images[imgid] = example["image"].convert("RGB")

    return images, captions


def get_text_embeds(encode_fn, captions_dict):
    """Mean of each image's caption embeddings, then L2-normalised."""
    embeds = []
    for caps in captions_dict.values():
        with torch.no_grad():
            cap_embeds = encode_fn(caps).cpu().numpy()
        embeds.append(l2_normalize(cap_embeds.mean(axis=0)))
    return np.asarray(embeds)


def get_img_embeds(encode_fn, images_dict):
    embeds = []
    for img in images_dict.values():
        with torch.no_grad():
            img_embed = encode_fn(img)
        embeds.append(l2_normalize(img_embed))
    return np.asarray(embeds)


def get_pairwise_cossim_mse(embeds_a, embeds_b):
    cos_sim_a = embeds_a.dot(embeds_a.T)
    cos_sim_b = embeds_b.dot(embeds_b.T)
    return ((cos_sim_a - cos_sim_b) ** 2).mean()


def main():
    train_ds, _, _ = load_datasets(data_dir=DEF_COCO_DIR, coco_dataset_ratio=2)
    images, captions = gather_images_and_captions(train_ds)

    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_text_embeds = get_text_embeds(lambda x: clip.encode_text(tokenizer(x).to(device)), captions)
    clip_img_embeds = get_img_embeds(
        lambda x: clip.encode_image(torch.stack([preprocess(x).to(device)])).cpu().numpy()[0], images)

    st_small = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    st_med = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device=device)
    st_s_embeds = get_text_embeds(lambda x: st_small.encode(x, convert_to_tensor=True), captions)
    st_m_embeds = get_text_embeds(lambda x: st_med.encode(x, convert_to_tensor=True), captions)

    # CLS token of the image-feature-extraction pipeline
    dinov3_s = pipeline(model="facebook/dinov3-vits16-pretrain-lvd1689m", task="image-feature-extraction")
    dinov3_b = pipeline(model="facebook/dinov3-vitb16-pretrain-lvd1689m", task="image-feature-extraction")
    dinov3_s_img_embeds = get_img_embeds(lambda x: dinov3_s(x)[0][0], images)
    dinov3_b_img_embeds = get_img_embeds(lambda x: dinov3_b(x)[0][0], images)

    # tab:rel_cosine_rmse
    base = get_pairwise_cossim_mse(st_m_embeds, st_s_embeds) ** 0.5
    print(f'Base, text vs text, raw value: {base}')
    for name, a, b in [("Vis vs vis", dinov3_b_img_embeds, dinov3_s_img_embeds),
                       ("Vis vs text", dinov3_b_img_embeds, st_m_embeds),
                       ("Vis vs CLIP vis", clip_img_embeds, dinov3_b_img_embeds),
                       ("Text vs CLIP text", clip_text_embeds, st_m_embeds),
                       ("CLIP vis vs CLIP text", clip_text_embeds, clip_img_embeds)]:
        print(f'{name}: {get_pairwise_cossim_mse(a, b) ** 0.5 / base}')

    # fraction of image pairs whose image and caption similarities differ by more than 0.4
    diffs = np.abs(dinov3_b_img_embeds.dot(dinov3_b_img_embeds.T) - st_m_embeds.dot(st_m_embeds.T))
    lower_tri_mask = np.tril(np.ones_like(diffs, dtype=bool), k=-1)
    print(f'Fraction of pairs with |image sim - caption sim| > 0.4: {(diffs[lower_tri_mask] > 0.4).mean()}')


if __name__ == '__main__':
    main()
