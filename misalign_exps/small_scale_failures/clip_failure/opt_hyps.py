import os
import random
import traceback
from datetime import datetime

import numpy as np
from transformers import AutoImageProcessor

from coco_utils import load_datasets, filter_and_preprocess_datasets
from small_scale_failures.clip_failure.clip_model_utils import CLIPDualEncoderModel
from small_scale_failures.clip_failure.misc_utils import get_parser, get_config
from small_scale_failures.clip_failure.training_utils import train_clip
from small_scale_failures.model_utils import get_tokenizer


def generate_hyperparameter_grid():
    """Generate hyperparameter combinations"""
    # Learning rates: 1e-5 to 1e-2 with 0.5 order of magnitude steps
    lrs = []
    for exp in np.arange(-5, -2 + 0.1, 0.5):  # -5, -4.5, -4, -3.5, -3, -2.5, -2
        lrs.append(10 ** exp)

    # Weight decays: 1e-4 to 1e-1 with 1 order of magnitude steps
    wds = []
    for exp in range(-4, -1 + 1):  # -4, -3, -2, -1
        wds.append(10 ** exp)

    # Create all combinations
    hyperparams = []
    for lr in lrs:
        for wd in wds:
            hyperparams.append((lr, wd))

    # Shuffle for random order
    random.shuffle(hyperparams)

    return hyperparams


def log_results(filepath, lr, wd, imagenet_top1, clip_score, is_best=False):
    """Log results to file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, 'a') as f:
        result_str = f"[{timestamp}] LR: {lr:.2e}, WD: {wd:.2e}, ImageNet Top-1: {imagenet_top1:.2f}%, CLIP Score: {clip_score:.4f}"
        if is_best:
            result_str += " *BEST*"
        f.write(result_str + "\n")
        f.flush()


def main():
    parser = get_parser()

    # Output arguments
    parser.add_argument("--base_out_dir", type=str, default="./hyperparam_search",
                        help="Base output directory for saving search results")
    parser.add_argument("--results_file", type=str, default="hyperparam_results.txt",
                        help="File to save hyperparameter search results")

    args = parser.parse_args()

    args.base_out_dir = os.path.join(args.base_out_dir, args.adapter_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.base_out_dir = os.path.join(args.base_out_dir, f"run_{timestamp}")

    # Create base output directory
    os.makedirs(args.base_out_dir, exist_ok=True)
    results_filepath = os.path.join(args.base_out_dir, args.results_file)

    # Log start of search
    with open(results_filepath, 'a') as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Hyperparameter search started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n\n")

    # Load and preprocess datasets once
    print("Loading datasets...")
    train_ds, valid_ds, test_ds = load_datasets(data_dir=args.data_dir, coco_dataset_ratio=args.coco_dataset_ratio)

    image_processor = AutoImageProcessor.from_pretrained(args.vis_enc)
    tokenizer = get_tokenizer(args.lang_enc)

    train_ds, valid_ds, test_ds = filter_and_preprocess_datasets(
        train_ds, valid_ds, test_ds,
        image_processor, tokenizer,
        max_length=32,
        append_eos=False,
    )

    del test_ds

    # Generate hyperparameter combinations
    hyperparams = generate_hyperparameter_grid()
    print(f"Testing {len(hyperparams)} hyperparameter combinations...")

    # Track best results - now based on ImageNet top-1
    best_imagenet_top1 = -float('inf')
    best_clip_score_at_best = -float('inf')
    best_lr = None
    best_wd = None

    # Run hyperparameter search
    for i, (lr, wd) in enumerate(hyperparams):
        print(f"\n[{i + 1}/{len(hyperparams)}] Testing LR={lr:.2e}, WD={wd:.2e}")

        # Create output directory for this run
        run_name = f"lr_{lr:.2e}_wd_{wd:.2e}"
        args.out_dir = os.path.join(args.base_out_dir, run_name)

        try:
            # Initialize model fresh for each run
            config = get_config(args)

            model = CLIPDualEncoderModel(config)

            # Set learning rate+wd for this run
            args.lr = lr
            args.wd = wd

            # Train with current hyperparameters
            trainer, best_val_imagenet_top1 = train_clip(model, args, train_ds, valid_ds, tokenizer, image_processor,
                                                         concept_dir=args.concepts_dir)

            model = trainer.model  # Get the best model

            val_res = trainer.evaluate(eval_dataset=valid_ds, metric_key_prefix="eval")
            imagenet_top1, clip_score = val_res['eval_imagenet_top1'], val_res['eval_clip_score']
            assert best_val_imagenet_top1 == imagenet_top1, \
                f"Validation metrics mismatch! Expected {best_val_imagenet_top1}, got {imagenet_top1}"

            # Get validation metrics
            imagenet_top1 = imagenet_top1 if imagenet_top1 is not None else -float('inf')

            is_best = imagenet_top1 > best_imagenet_top1
            if is_best:
                best_imagenet_top1 = imagenet_top1
                best_clip_score_at_best = clip_score
                best_lr = lr
                best_wd = wd

                # Save the best model
                model.save_pretrained(os.path.join(args.base_out_dir, "best_model"))

            # Log results
            log_results(results_filepath, lr, wd, imagenet_top1, clip_score, is_best)
            print(f"ImageNet Top-1: {imagenet_top1:.2f}%, CLIP Score: {clip_score:.4f} {'*BEST*' if is_best else ''}")

        except Exception as e:
            traceback.print_exc()
            print(f"Error during training: {e}")
            log_results(results_filepath, lr, wd, -1.0, -1.0)  # Log failed run
            continue

    # Log final results
    with open(results_filepath, 'a') as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Search completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best hyperparameters:\n")
        f.write(f"  Learning Rate: {best_lr:.2e}\n")
        f.write(f"  Weight Decay: {best_wd:.2e}\n")
        f.write(f"  ImageNet Top-1: {best_imagenet_top1:.2f}%\n")
        f.write(f"  CLIP Score (at best): {best_clip_score_at_best:.4f}\n")
        f.write(f"  Model saved at: {os.path.join(args.base_out_dir, 'best_model')}\n")
        f.write(f"{'=' * 60}\n")

    print(
        f"\nSearch complete! Best LR={best_lr:.2e}, WD={best_wd:.2e}, ImageNet Top-1={best_imagenet_top1:.2f}%, CLIP Score={best_clip_score_at_best:.4f}")


if __name__ == '__main__':
    main()
