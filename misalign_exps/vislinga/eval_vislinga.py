# SPDX-License-Identifier: MIT
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


def parse_openclip(name: str) -> tuple[str, str] | None:
    if name.startswith('openclip:'):
        _, arch, pretrained = name.split(':')
        return arch, pretrained
    return None


def embed_text(model_name: str, words: list[str], device: str) -> tuple[np.ndarray, int]:
    oc = parse_openclip(model_name)
    if oc is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device, model_kwargs={'torch_dtype': 'auto'})
        n_params = sum(p.numel() for p in model.parameters())
        embeds = model.encode(words, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
        return embeds.astype(np.float32), n_params

    import open_clip
    arch, pretrained = oc
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model = model.to(device).eval()  # OpenCLIP has no 'auto' dtype; keeps the checkpoint's precision
    tokenizer = open_clip.get_tokenizer(arch)
    n_params = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.visual.parameters())
    out = []
    with torch.no_grad():
        for i in range(0, len(words), 256):
            feats = model.encode_text(tokenizer(words[i:i + 256]).to(device)).float()
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
    return np.concatenate(out), n_params


def embed_images(model_name: str, images, concepts: list[str], device: str,
                 batch_size: int) -> tuple[np.ndarray, int]:
    """Embed every image, then mean-pool per concept (unnormalised mean, then L2-normalised, as in the paper)."""
    oc = parse_openclip(model_name)
    if oc is None:
        from transformers import pipeline
        pipe = pipeline(task='image-feature-extraction', model=model_name, device=device, torch_dtype='auto')
        n_params = sum(p.numel() for p in pipe.model.parameters())

        def encode(batch):
            return np.stack([np.asarray(o, dtype=np.float32)[0, 0] for o in pipe(batch, batch_size=len(batch))])  # CLS
    else:
        import open_clip
        arch, pretrained = oc
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model = model.to(device).eval()  # OpenCLIP has no 'auto' dtype; keeps the checkpoint's precision
        n_params = sum(p.numel() for p in model.visual.parameters())

        def encode(batch):
            x = torch.stack([preprocess(im) for im in batch]).to(device)
            with torch.no_grad():
                return model.encode_image(x).float().cpu().numpy()

    sums, counts = {}, {}
    for i in tqdm(range(0, len(images), batch_size), desc='images'):
        rows = images[i:i + batch_size]
        feats = encode([im.convert('RGB') for im in rows['image']])
        for concept, f in zip(rows['concept'], feats):
            sums[concept] = sums.get(concept, 0) + f
            counts[concept] = counts.get(concept, 0) + 1
    means = np.stack([sums[c] / counts[c] for c in concepts])
    return means / np.linalg.norm(means, axis=1, keepdims=True), n_params


def score(embeds: np.ndarray, concepts: list[str], analogies) -> dict:
    idx = {c: i for i, c in enumerate(concepts)}
    a, b, c, d = (np.array([idx[w] for w in analogies[f'word{k}']]) for k in range(1, 5))
    target = embeds[b] - embeds[a] + embeds[c]
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    sims = target @ embeds.T  # (n_analogies, n_concepts)
    rows = np.arange(len(d))
    sim_d = sims[rows, d]
    for k in (a, b, c):  # A, B and C are not candidates
        sims[rows, k] = -np.inf
    ranks = 1 + (sims > sim_d[:, None]).sum(axis=1)

    categories = np.array(analogies['category'])
    per_cat = {cat: {'n': int((m := categories == cat).sum()), 'top10': float(np.mean(ranks[m] <= 10)),
                     'avg_rank': float(ranks[m].mean())}
               for cat in sorted(set(categories))}
    return {'scores': {'top1': float(np.mean(ranks == 1)), 'top10': float(np.mean(ranks <= 10)),
                       'avg_rank': float(ranks.mean()), 'median_rank': float(np.median(ranks))},
            'per_category': per_cat}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help="HF id, sentence-transformers id or 'openclip:<arch>:<pretrained>'")
    p.add_argument('--modality', choices=['text', 'image'], required=True)
    p.add_argument('--data', default='YGideoni/VisLingA', help='HF repo id or local dataset directory')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()

    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analogies = load_dataset(args.data, 'analogies', split='test')
    concepts = sorted(
        set(analogies['word1']) | set(analogies['word2']) | set(analogies['word3']) | set(analogies['word4']))

    if args.modality == 'text':
        embeds, n_params = embed_text(args.model, concepts, device)
    else:
        images = load_dataset(args.data, 'images', split='test')
        missing = set(concepts) - set(images['concept'])
        assert not missing, f'concepts without images: {sorted(missing)[:10]}'
        embeds, n_params = embed_images(args.model, images, concepts, device, args.batch_size)

    result = {'model': args.model, 'modality': args.modality, 'n_params': n_params,
              'data': args.data, 'n_concepts': len(concepts), 'n_analogies': len(analogies),
              'seconds': round(time.time() - t0, 1), **score(np.asarray(embeds, dtype=np.float32), concepts, analogies)}
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"{args.modality}__{args.model.replace('/', '__').replace(':', '__')}.json"
    out_file.write_text(json.dumps(result, indent=1))
    r = result['scores']
    print(f"{args.model} [{args.modality}] top10={r['top10']:.4f} top1={r['top1']:.4f} avg_rank={r['avg_rank']:.1f} "
          f"-> {out_file}")


if __name__ == '__main__':
    main()
