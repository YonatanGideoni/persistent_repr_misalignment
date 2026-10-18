import urllib.request
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from constants import get_concepts, device


class ConceptDataset(Dataset):
    """Dataset for loading concept images."""

    def __init__(self, root_dir, image_processor, concepts=None):
        if concepts is None:
            concepts = get_concepts()

        self.root_dir = Path(root_dir)
        self.image_processor = image_processor
        self.samples = []

        # Collect all images with their concept labels
        for concept in concepts:
            concept_dir = self.root_dir / concept
            if concept_dir.exists():
                for img_path in concept_dir.glob('*'):
                    self.samples.append({
                        'path': str(img_path),
                        'concept': concept,
                        'image': Image.open(img_path).convert('RGB')
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        no_image_proc = self.image_processor is None

        if no_image_proc:
            return {
                'image': sample['image'],
                'concept': sample['concept'],
                'path': sample['path']
            }

        image = (
            self.image_processor(sample['image'])
            if isinstance(self.image_processor, T.Compose)
            else self.image_processor(images=sample['image'], return_tensors="pt").pixel_values.squeeze(0)
        )
        return {
            'pixel_values': image,
            'concept': sample['concept'],
            'path': sample['path']
        }


def concept_prepare_text_prompts(imagenet_classes, concepts_to_eval):
    """
    Prepare text prompts for concepts and ImageNet distractors.

    Args:
        imagenet_classes: List of ImageNet class names
        concepts_to_eval: List of concepts to evaluate

    Returns:
        all_prompts: List of all text prompts
        concept_indices: Dict mapping concept names to their indices in all_prompts
    """
    if concepts_to_eval is None:
        concepts_to_eval = get_concepts()

    # Create prompts for concepts
    concept_prompts = [f"a photo of a {concept}" for concept in concepts_to_eval]

    # Filter out ImageNet classes that overlap with our concepts
    # Also filter out very similar words (e.g., "umbrellas" vs "umbrella")
    overlapping_concepts = set()
    for concept in concepts_to_eval:
        overlapping_concepts.add(concept)
        overlapping_concepts.add(concept + 's')  # plural form
        overlapping_concepts.add(concept.rstrip('s'))  # singular form if concept is plural

    distractor_classes = []
    for cls in imagenet_classes:
        # Check if the class name contains any of our concepts
        if not any(concept in cls.lower() for concept in overlapping_concepts):
            distractor_classes.append(cls)

    distractor_prompts = [f"a photo of a {cls}" for cls in distractor_classes]

    # Combine all prompts
    all_prompts = concept_prompts + distractor_prompts

    # Create indices mapping
    concept_indices = {concept: i for i, concept in enumerate(concepts_to_eval)}

    print(f'Using {len(concept_indices)} concepts and {len(distractor_prompts)} distractors')

    return all_prompts, concept_indices


def concept_compute_text_features(model, tokenizer, prompts, batch_size=100):
    """
    Compute text features for all prompts.

    Args:
        model: CLIP model
        tokenizer: Text tokenizer
        prompts: List of text prompts
        batch_size: Batch size for processing

    Returns:
        text_features: Normalized text features tensor
    """
    model.eval()
    text_features_list = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=77
            )

            # Get text features
            text_embeddings = model.get_text_features(
                inputs['input_ids'].to(device),
                inputs['attention_mask'].to(device)
            )

            # Normalize
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
            text_features_list.append(text_embeddings)

    return torch.cat(text_features_list, dim=0)


def evaluate_concept_detection_given_text_feats(model, dataloader, text_features, concept_indices, concepts_to_eval):
    """
    Evaluate concept detection performance.

    Args:
        model: CLIP model
        dataloader: DataLoader for concept images
        text_features: Pre-computed normalized text features
        concept_indices: Dict mapping concept names to indices
        concepts_to_eval: List of concepts being evaluated

    Returns:
        results: Dict with evaluation results
    """
    model.eval()

    # Track predictions for both top-1 and top-5
    correct_predictions_top1 = defaultdict(int)
    correct_predictions_top5 = defaultdict(int)
    total_predictions = defaultdict(int)
    all_correct_top1 = 0
    all_correct_top5 = 0
    all_total = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            concepts = batch['concept']

            # Get image features
            vision_embeddings = model.get_vision_features(pixel_values)
            vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)

            # Compute similarities with all text prompts
            similarities = torch.matmul(vision_embeddings, text_features.t())

            # Get top-5 predictions
            _, top5_indices = similarities.topk(5, dim=1)

            # Check if predictions are correct
            for concept, top5_preds in zip(concepts, top5_indices):
                correct_idx = concept_indices[concept]

                # Check top-1 accuracy
                is_correct_top1 = (top5_preds[0].item() == correct_idx)

                # Check top-5 accuracy
                is_correct_top5 = correct_idx in top5_preds.tolist()

                # Update counters
                if is_correct_top1:
                    correct_predictions_top1[concept] += 1
                    all_correct_top1 += 1

                if is_correct_top5:
                    correct_predictions_top5[concept] += 1
                    all_correct_top5 += 1

                total_predictions[concept] += 1
                all_total += 1

    # Calculate accuracies
    per_concept_accuracy = {}
    for concept in concepts_to_eval:
        if total_predictions[concept] > 0:
            accuracy_top1 = correct_predictions_top1[concept] / total_predictions[concept]
            accuracy_top5 = correct_predictions_top5[concept] / total_predictions[concept]
            per_concept_accuracy[concept] = {
                'accuracy_top1': accuracy_top1,
                'accuracy_top5': accuracy_top5,
                'correct_top1': correct_predictions_top1[concept],
                'correct_top5': correct_predictions_top5[concept],
                'total': total_predictions[concept]
            }

    overall_accuracy_top1 = all_correct_top1 / all_total if all_total > 0 else 0
    overall_accuracy_top5 = all_correct_top5 / all_total if all_total > 0 else 0

    results = {
        'per_concept_accuracy': per_concept_accuracy,
        'overall_accuracy_top1': overall_accuracy_top1,
        'overall_accuracy_top5': overall_accuracy_top5,
        'total_correct_top1': all_correct_top1,
        'total_correct_top5': all_correct_top5,
        'total_images': all_total,
    }

    return results


def pil_list_collate_fn(batch):
    images = [item['image'] for item in batch]
    concepts = [item['concept'] for item in batch]
    paths = [item['path'] for item in batch]
    return {'image': images, 'concept': concepts, 'path': paths}


def setup_concept_detection_data(root_dir, image_processor, batch_size=32, concepts=None):
    dataset = ConceptDataset(root_dir, image_processor, concepts)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=pil_list_collate_fn if image_processor is None else None
    )

    # Get ImageNet classes and prepare prompts
    imagenet_classes = get_imagenet_class_names()
    all_prompts, concept_indices = concept_prepare_text_prompts(imagenet_classes, concepts)

    return dataloader, all_prompts, concept_indices


def eval_concept_detection(model, tokenizer, all_prompts, concept_indices, dataloader):
    text_features = concept_compute_text_features(model, tokenizer, all_prompts)

    concepts = get_concepts()
    results = evaluate_concept_detection_given_text_feats(model, dataloader, text_features, concept_indices, concepts)
    return results


def get_imagenet_class_names():
    """Get ImageNet class names from the standard synset labels."""
    # Download the class index to name mapping used by PyTorch/torchvision
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

    with urllib.request.urlopen(url) as response:
        class_names = [line.decode('utf-8').strip() for line in response.readlines()]
    return class_names


def compute_imagenet_accuracy(model, imagenet_loader, encoded_prompts):
    """Compute top-1 accuracy on ImageNet validation set."""
    model.eval()

    # Compute text features (fresh each time since language model is being trained)
    with torch.no_grad():
        text_features_list = []
        batch_size = 1000

        for i in range(0, len(encoded_prompts['input_ids']), batch_size):
            batch_input_ids = encoded_prompts['input_ids'][i:i + batch_size].to(device)
            batch_attention_mask = encoded_prompts['attention_mask'][i:i + batch_size].to(device)

            text_embeddings = model.get_text_features(batch_input_ids, batch_attention_mask)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
            text_features_list.append(text_embeddings)

        text_features = torch.cat(text_features_list, dim=0)  # [1000, embed_dim]

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in imagenet_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get image features
            vision_embeddings = model.get_vision_features(images)
            vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)

            # Compute similarities with all class embeddings
            similarities = torch.matmul(vision_embeddings, text_features.t())

            # Get predictions
            _, predicted = similarities.max(1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy
