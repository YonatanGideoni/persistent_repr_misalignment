import hashlib
import json
import os
import pickle

import aiohttp
import numpy as np
import open_clip
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import get_concepts, device


def get_conceptless_data(dataset, tokenizer):
    sents = tokenizer.batch_decode(dataset['labels'], skip_special_tokens=True)
    concepts = get_concepts()
    indices_to_keep = [i for i, sent in enumerate(sents) if all(concept not in sent for concept in concepts)]

    return dataset.select(indices_to_keep)


def load_datasets(data_dir, coco_dataset_ratio):
    stor_opts = {'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}  # needed to prevent timeout errs
    train_ds = load_dataset("HuggingFaceM4/COCO", split=f"train[:{coco_dataset_ratio}%]", cache_dir=data_dir,
                            trust_remote_code=True, storage_options=stor_opts)
    valid_ds = load_dataset("HuggingFaceM4/COCO", split=f"validation[:{coco_dataset_ratio}%]", cache_dir=data_dir,
                            trust_remote_code=True, storage_options=stor_opts)
    test_ds = load_dataset("HuggingFaceM4/COCO", split="test", cache_dir=data_dir, trust_remote_code=True,
                           storage_options=stor_opts)
    return train_ds, valid_ds, test_ds


def filter_and_preprocess_dataset(ds, image_processor, tokenizer, max_length, append_eos=True):
    # Filter out invalid images first
    ds = ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=16)

    # Original preprocessing for individual captions
    def preprocess(items):
        pixel_values = image_processor(items["image"], return_tensors="pt").pixel_values.squeeze(0)

        # Get original caption
        sentence = items["sentences"]["raw"].lower().replace('.', '')
        if append_eos:
            sentence += tokenizer.eos_token

        targets = tokenizer(sentence, max_length=max_length, padding="max_length", truncation=True,
                            return_tensors="pt")
        return dict(pixel_values=pixel_values, labels=targets["input_ids"].squeeze(0))

    ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=16)
    ds.set_format(type='torch', columns=['pixel_values', 'labels'])

    return ds


def filter_and_preprocess_datasets(train_ds, valid_ds, test_ds, image_processor, tokenizer, max_length, append_eos=True):
    train_ds = filter_and_preprocess_dataset(train_ds, image_processor, tokenizer, max_length, append_eos)
    valid_ds = filter_and_preprocess_dataset(valid_ds, image_processor, tokenizer, max_length, append_eos)
    test_ds = filter_and_preprocess_dataset(test_ds, image_processor, tokenizer, max_length, append_eos)

    return train_ds, valid_ds, test_ds


def create_dataset_fingerprint(dataset, concepts, batch_size, sim_threshold):
    """Create a unique fingerprint for the dataset and filtering parameters."""
    dataset_info = {
        'dataset_class': dataset.__class__.__name__,
        'length': len(dataset),
        'batch_size': batch_size,
        'sim_threshold': sim_threshold,
        'concepts': concepts,
    }

    # Try to get built-in dataset identifiers
    try:
        if hasattr(dataset, 'name'):
            dataset_info['name'] = dataset.name
        if hasattr(dataset, '_fingerprint'):
            dataset_info['fingerprint'] = dataset._fingerprint
    except:
        pass

    # Sample items for structure fingerprinting
    try:
        sample_count = min(5, len(dataset))
        step = max(1, len(dataset) // sample_count)
        sample_indices = [i * step for i in range(sample_count)]

        sample_items = []
        for idx in sample_indices:
            item = dataset[idx]
            sample_item_info = {}
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    sample_item_info[key + '_shape'] = list(value.shape)
                elif hasattr(value, 'shape'):
                    sample_item_info[key + '_shape'] = list(value.shape)
                else:
                    sample_item_info[key + '_type'] = str(type(value))
            sample_items.append(sample_item_info)
        dataset_info['samples'] = sample_items
    except Exception as e:
        print(f"Warning: Could not sample dataset items: {e}")

    # Create hash from dataset info
    dataset_str = json.dumps(dataset_info, sort_keys=True)
    return hashlib.md5(dataset_str.encode()).hexdigest()


def get_cache_path(dataset_hash, cache_dir='./clip_filter_cache'):
    """Get the path to the cache file for the given dataset hash."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{dataset_hash}.pkl')


def load_cached_indices(cache_file):
    """Load indices from cache file if it exists."""
    if os.path.exists(cache_file):
        print(f"Loading cached filtered indices from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def save_indices_to_cache(indices, cache_file):
    """Save indices to cache file."""
    print(f"Caching {len(indices)} filtered indices to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(indices, f)


def filter_dataset_with_clip(dataset, concepts, batch_size=128, sim_threshold=0.2):
    """Filter the dataset using CLIP to remove images containing specified concepts."""
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model = model.to(device)

    model.eval()
    clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')

    # Encode text queries
    concepts_texts = clip_tokenizer([f"a photo of a {concept}" for concept in concepts]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(concepts_texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    indices_to_keep = []
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Filtering dataset"):
            batch_images = batch['pixel_values'].to(device, non_blocking=True)

            image_features = model.encode_image(batch_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).max(dim=-1).values

            # Convert to CPU and numpy to free GPU memory
            batch_indices = (similarity < sim_threshold).cpu().numpy()
            indices_to_keep.extend([i * batch_size + idx for idx, keep in enumerate(batch_indices) if keep])

    return indices_to_keep


def get_conceptless_data_sim_filter(dataset, batch_size=128, sim_threshold=0.2, cache_dir='./clip_filter_cache'):
    """
    Filter a dataset to remove images that contain specified concepts, with disk caching.

    Args:
        dataset: The dataset to filter
        batch_size: Batch size for processing images
        sim_threshold: Similarity threshold below which to keep images
        cache_dir: Directory to store cache files

    Returns:
        Filtered dataset
    """
    CONCEPTS = get_concepts()

    # Create fingerprint and get cache file path
    dataset_hash = create_dataset_fingerprint(dataset, CONCEPTS, batch_size, sim_threshold)
    cache_file = get_cache_path(dataset_hash, cache_dir)

    # Try to load from cache
    indices_to_keep = load_cached_indices(cache_file)

    # If no cache, compute the indices
    if indices_to_keep is None:
        print(f"Computing indices for dataset of length {len(dataset)}")
        indices_to_keep = filter_dataset_with_clip(dataset, CONCEPTS, batch_size, sim_threshold)
        save_indices_to_cache(indices_to_keep, cache_file)

    # Return filtered dataset
    return dataset.select(indices_to_keep)

