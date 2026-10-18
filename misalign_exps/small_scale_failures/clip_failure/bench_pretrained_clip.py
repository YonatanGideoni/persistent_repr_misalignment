import argparse
import json
from pathlib import Path

from transformers import CLIPModel, CLIPProcessor

from constants import device, IMG_CONCEPTS_DIR
from small_scale_failures.clip_failure.bench_utils import setup_concept_detection_data, eval_concept_detection

CLIPModel.get_vision_features = CLIPModel.get_image_features


def print_results(results):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 80)
    print("CLIP Concept Detection Results")
    print("=" * 80)
    print("\nPer-Concept Accuracy:")
    print("-" * 70)
    print(f"{'Concept':<15} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Top-1':<12} {'Top-5':<12}")
    print(f"{'':15} {'':12} {'':12} {'Correct/Total':<12} {'Correct/Total':<12}")
    print("-" * 70)
    for concept, stats in sorted(results['per_concept_accuracy'].items()):
        accuracy_top1, accuracy_top5 = stats['accuracy_top1'], stats['accuracy_top5']
        correct_top1, correct_top5 = stats['correct_top1'], stats['correct_top5']
        total = stats['total']
        print(f"{concept:<15} {accuracy_top1:>10.1%}   {accuracy_top5:>10.1%}   "
              f"{correct_top1:>3}/{total:<8} {correct_top5:>3}/{total:<3}")
    print("-" * 70)
    print(f"\nOverall Top-1 Accuracy: {results['overall_accuracy_top1']:.1%} "
          f"({results['total_correct_top1']}/{results['total_images']})")
    print(f"Overall Top-5 Accuracy: {results['overall_accuracy_top5']:.1%} "
          f"({results['total_correct_top5']}/{results['total_images']})")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark standard OpenAI CLIP on concept detection')
    parser.add_argument('--root_dir', type=str, default=IMG_CONCEPTS_DIR,
                        help='Directory containing concept images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--save_detailed', type=str, default=None,
                        help='Path to save detailed results as a JSON file (optional)')
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory {root_dir} does not exist")

    # Hardcode the model name for standard CLIP evaluation
    model_name = "openai/clip-vit-base-patch16"
    print(f"Loading standard CLIP model: {model_name}...")

    # Load model and processor directly from Hugging Face Hub
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # The processor conveniently contains both the image_processor and tokenizer
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

    # Prepare dataset using the existing utility function
    dataloader, all_prompts, concept_indices = setup_concept_detection_data(
        root_dir, image_processor, batch_size=args.batch_size
    )

    # Evaluate using the existing utility function
    print("Evaluating concept detection on standard CLIP...")
    results = eval_concept_detection(model, tokenizer, all_prompts, concept_indices, dataloader)

    # Print results using the existing utility function
    print_results(results)

    # Save detailed results if requested
    if args.save_detailed:
        # Correctly saving the metrics available in the results dictionary
        save_path = Path(args.save_detailed)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_results = {
            'model_name': model_name,
            'per_concept_accuracy': results['per_concept_accuracy'],
            'overall_accuracy_top1': results['overall_accuracy_top1'],
            'overall_accuracy_top5': results['overall_accuracy_top5'],
            'total_correct_top1': results['total_correct_top1'],
            'total_correct_top5': results['total_correct_top5'],
            'total_images': results['total_images'],
        }
        with open(save_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nDetailed results saved to {args.save_detailed}")


if __name__ == '__main__':
    main()
