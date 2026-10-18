import argparse
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from llava.constants import IMAGE_PLACEHOLDER, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, \
    IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from llava.model.builder import load_pretrained_model


class ImageCaptioningDataset(Dataset):
    def __init__(self, root_dir, valid_extensions=('.jpg', '.jpeg', '.png')):
        self.root_dir = Path(root_dir)

        # Get all image paths and their corresponding directory names
        self.image_files = []
        self.target_words = []

        for dir_path in self.root_dir.iterdir():
            if dir_path.is_dir():
                target_word = dir_path.name.lower()
                for img_path in dir_path.glob('*'):
                    if img_path.suffix.lower() in valid_extensions:
                        self.image_files.append(img_path)
                        self.target_words.append(target_word)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return {
            'image': image,
            'path': str(image_path),
            'target_word': self.target_words[idx]
        }


def collate_fn(batch):
    images = [item['image'] for item in batch]
    paths = [item['path'] for item in batch]
    target_words = [item['target_word'] for item in batch]

    return {
        'images': images,
        'paths': paths,
        'target_words': target_words
    }


def print_concept_statistics(results):
    concept_stats = defaultdict(lambda: {'total': 0, 'detected': 0, 'examples': [], 'positive_examples': []})

    for r in results:
        stats = concept_stats[r['target_word']]
        stats['total'] += 1
        if r['detected']:
            stats['detected'] += 1
            stats['positive_examples'].append((r['path'], r['caption']))
        else:
            stats['examples'].append((r['path'], r['caption']))

    print("\nPer-Concept Statistics:")
    print("-" * 60)
    print(f"{'Concept':<15} {'Detection Rate':<15} {'Detected/Total':<15}")
    print("-" * 60)

    overall_detected = 0
    overall_total = 0

    sorted_concepts = sorted(
        concept_stats.items(),
        key=lambda x: x[1]['detected'] / x[1]['total'] if x[1]['total'] > 0 else 0,
        reverse=True
    )

    for concept, stats in sorted_concepts:
        detection_rate = stats['detected'] / stats['total']
        print(f"{concept:<15} {detection_rate:>13.1%}  {stats['detected']}/{stats['total']:<10}")
        overall_detected += stats['detected']
        overall_total += stats['total']

    print("-" * 60)
    print(f"{'Overall':<15} {overall_detected / overall_total:>13.1%}  {overall_detected}/{overall_total}")

    print("\nFailed Detection Examples:")
    print("=" * 80)
    for concept, stats in sorted_concepts:
        if len(stats['examples']) > 0:
            print(f"\n{concept.upper()}:")
            for path, caption in stats['examples'][:15]:
                print(f"  Path: {path}")
                print(f"  Caption: {caption}\n")

    print("\nSuccessful Detection Examples:")
    print("=" * 80)
    for concept, stats in sorted_concepts:
        if stats['positive_examples']:
            print(f"\n{concept.upper()}:")
            for path, caption in stats['positive_examples'][:15]:
                print(f"  Path: {path}")
                print(f"  Caption: {caption}\n")


def check_captions_and_save(model, image_processor, tokenizer, root_dir, prompt,
                            get_conv_prompt, process_images, tokenizer_image_token,
                            IMAGE_TOKEN_INDEX, output_dir, num_batches_to_save=3,
                            batch_size=32, max_new_tokens=64):
    """
    Process images and save the first few batches with their captions.

    Args:
        output_dir: Directory to save images and captions
        num_batches_to_save: Number of batches to save (default: 3)
        max_new_tokens: Generation length limit (default: 64)
        Other args same as check_captions
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Create caption log file
    caption_log = run_dir / "captions.txt"

    dataset = ImageCaptioningDataset(root_dir)

    subset_size = int(1 * len(dataset))
    indices = random.sample(range(len(dataset)), subset_size)
    subset = Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )

    results = []
    batches_saved = 0
    total_images_saved = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing images")):
            # Process batch of images
            images = batch['images']
            image_sizes = [img.size for img in images]

            # Convert batch of images to tensor
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)

            # Prepare prompt
            conv_prompt = get_conv_prompt(prompt)
            input_ids = (
                tokenizer_image_token(conv_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            input_ids = torch.stack([input_ids] * len(images), dim=0).squeeze()

            # Generate outputs for batch
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

            captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # Save images and captions for the first few batches
            if batches_saved < num_batches_to_save:
                batch_dir = run_dir / f"batch_{batch_idx:03d}"
                batch_dir.mkdir(exist_ok=True)

                with open(caption_log, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'=' * 80}\n")
                    f.write(f"BATCH {batch_idx}\n")
                    f.write(f"{'=' * 80}\n\n")

                for img_idx, (image, path, caption, target_word) in enumerate(
                    zip(images, batch['paths'], captions, batch['target_words'])
                ):
                    # Create filename with index and target word
                    img_filename = f"{img_idx:03d}_{target_word}_{Path(path).stem}.jpg"
                    img_path = batch_dir / img_filename

                    # Save image
                    image.save(img_path, "JPEG", quality=95)

                    # Write caption to log file
                    with open(caption_log, 'a', encoding='utf-8') as f:
                        f.write(f"Image: {img_filename}\n")
                        f.write(f"Original path: {path}\n")
                        f.write(f"Target word: {target_word}\n")
                        f.write(f"Caption: {caption}\n")
                        f.write(f"Detected: {'YES' if target_word.lower() in caption.lower() else 'NO'}\n")
                        f.write(f"{'-' * 40}\n\n")

                    total_images_saved += 1

                batches_saved += 1
                print(f"Saved batch {batch_idx} ({len(images)} images) to {batch_dir}")

            # Process results as before
            for path, caption, target_word in zip(batch['paths'], captions, batch['target_words']):
                caption = caption.lower()
                results.append({
                    'path': path,
                    'caption': caption,
                    'target_word': target_word,
                    'detected': target_word in caption
                })

    print(f"\nSaved {total_images_saved} images across {batches_saved} batches to {run_dir}")
    print(f"Caption log saved to {caption_log}")

    return results


def get_conv_prompt(qs):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
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


DEFAULT_ROOT_DIR = str(Path(__file__).resolve().parent.parent / 'concept_testset' / 'img_concepts_outs')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_paths', nargs='+', required=True,
                        help='One or more LLaVA checkpoints (local LoRA dirs or HF repo ids), evaluated in turn.')
    parser.add_argument('--model_base', default='lmsys/vicuna-7b-v1.5',
                        help='Base LLM the LoRA checkpoints were trained on.')
    parser.add_argument('--root_dir', default=DEFAULT_ROOT_DIR,
                        help='Concept test set: one sub-directory of images per concept '
                             '(default: ../concept_testset/img_concepts_outs relative to this repo).')
    parser.add_argument('--prompt', default='What is in this image?',
                        help='Question asked for every image. The appendix short-answer variant used '
                             '"Without saying anything else, please state the main thing in this image. '
                             'For example: dog." with --max_new_tokens 8')
    parser.add_argument('--max_new_tokens', type=int, default=64,
                        help='Generation length limit: 64 for the main "What is in this image?" eval, '
                             '8 for the appendix short-answer variant.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', default='./bench_output',
                        help='Where saved image/caption batches go (only if --num_batches_to_save > 0).')
    parser.add_argument('--num_batches_to_save', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_base = args.model_base

    for model_path in args.model_paths:
        print(f"\n\n=== Evaluating model at {model_path} ===\n")

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                               device_map='cuda')

        model.to('cuda')
        model.half()

        results = check_captions_and_save(
            model=model,
            image_processor=image_processor,
            tokenizer=tokenizer,
            root_dir=args.root_dir,
            prompt=args.prompt,
            get_conv_prompt=get_conv_prompt,
            process_images=process_images,
            tokenizer_image_token=tokenizer_image_token,
            IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            num_batches_to_save=args.num_batches_to_save,
            max_new_tokens=args.max_new_tokens
        )

        print_concept_statistics(results)

        del model, image_processor, tokenizer
        torch.cuda.empty_cache()
