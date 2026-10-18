import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
from llava.train.llava_trainer import LLaVATrainer


def get_conv_prompt(qs, model_name, model_config):
    """Get conversation prompt based on model type."""
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model_config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model_config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    conv_prompt = conv.get_prompt()

    return conv_prompt


class ConceptDetectionDataset(Dataset):
    """Dataset for concept detection evaluation during training."""

    def __init__(self,
                 root_dir: str,
                 tokenizer,
                 data_args,
                 model_name: str,
                 model_config,
                 valid_extensions=('.jpg', '.jpeg', '.png')):
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_name = model_name
        self.model_config = model_config

        # Get all image paths and their corresponding concept names
        self.samples = []

        for concept_dir in sorted(self.root_dir.iterdir()):
            if concept_dir.is_dir():
                concept_name = concept_dir.name.lower()
                for img_path in concept_dir.glob('*'):
                    if img_path.suffix.lower() in valid_extensions:
                        self.samples.append({
                            'image_path': str(img_path),
                            'concept': concept_name,
                            'id': f"{concept_name}_{img_path.stem}"
                        })

        print(
            f"Loaded {len(self.samples)} evaluation images across {len(set(s['concept'] for s in self.samples))} concepts")

        # Create the evaluation prompt in conversation format
        self.eval_prompt = "What is in this image?"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image - return PIL image for processing later
        image = Image.open(sample['image_path']).convert('RGB')

        # Get conversation prompt using the proper template
        conv_prompt = get_conv_prompt(self.eval_prompt, self.model_name, self.model_config)

        # Tokenize
        input_ids = tokenizer_image_token(
            conv_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        )

        return {
            'input_ids': input_ids,
            'labels': torch.zeros_like(input_ids),  # Dummy labels
            'image': image,
            'concept': sample['concept'],
            'id': sample['id']
        }


class ConceptDetectionCollator:
    """Custom collator for concept detection that preserves concept information."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        # Extract components
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        images = [instance['image'] for instance in instances]
        concepts = [instance['concept'] for instance in instances]
        ids = [instance['id'] for instance in instances]

        # Pad input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100  # IGNORE_INDEX
        )

        # Limit to model max length
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'images': images,  # Keep as PIL images
            'concepts': concepts,
            'ids': ids
        }

        return batch


def compute_metrics_wrapper(trainer, eval_dataset):
    """
    Wrapper function to properly handle generation-based evaluation.
    """
    model = trainer.model
    tokenizer = trainer.tokenizer

    # Get image processor from model
    image_processor = model.get_vision_tower().image_processor

    # Ensure model is in eval mode
    model.eval()

    # Create dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=ConceptDetectionCollator(tokenizer)
    )

    all_concepts = []
    all_detections = []

    generation_config = {
        'do_sample': False,
        'temperature': 0,
        'top_p': None,
        'num_beams': 1,
        'max_new_tokens': 64,
        'use_cache': True,
    }

    with torch.inference_mode():
        for batch in tqdm(eval_dataloader, desc="Evaluating concepts"):
            # Move batch to device
            input_ids = batch['input_ids'].to(model.device)

            # Process images
            images = batch['images']
            image_sizes = [img.size for img in images]

            model_dtype = next(model.parameters()).dtype
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=model_dtype)

            # Generate captions
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                **generation_config
            )

            # Decode outputs
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # Extract generated text
            for output, concept in zip(outputs, batch['concepts']):
                caption = output.lower()

                detected = concept in caption

                all_concepts.append(concept)
                all_detections.append(detected)

    # Calculate metrics
    metrics = {}

    # Per-concept accuracy
    for concept in set(all_concepts):
        concept_detections = [d for c, d in zip(all_concepts, all_detections) if c == concept]
        accuracy = sum(concept_detections) / len(concept_detections) if concept_detections else 0
        metrics[f'{concept}_acc'] = accuracy

    # Overall accuracy
    overall_accuracy = sum(all_detections) / len(all_detections) if all_detections else 0
    metrics['overall_concept_acc'] = overall_accuracy

    return metrics


class LLaVATrainerWithConceptEval(LLaVATrainer):
    """Custom trainer that supports concept detection evaluation."""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to handle generation-based concept detection."""

        # Check if we have a concept detection dataset
        eval_dataset = eval_dataset or self.eval_dataset

        if eval_dataset is not None and isinstance(eval_dataset, ConceptDetectionDataset):
            # Use our custom evaluation
            metrics = compute_metrics_wrapper(self, eval_dataset)

            # Add prefix to metrics
            metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

            # Log metrics
            self.log(metrics)

            # Call callbacks
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics
            )

            return metrics
        else:
            # Fall back to default evaluation
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )
