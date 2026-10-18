import argparse
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from constants import device, IMG_CONCEPTS_DIR
from small_scale_failures.clip_failure.bench_utils import ConceptDataset
from small_scale_failures.generation_failure.bench_utils import eval_concept_detection


def main():
    parser = argparse.ArgumentParser(description='Evaluate pretrained VIT-GPT2 on concept detection')
    parser.add_argument('--concept_dir', type=str, default=IMG_CONCEPTS_DIR, 
                        help=f'Path to concept images directory (default: {IMG_CONCEPTS_DIR})')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    args = parser.parse_args()

    # Load pretrained model and processors
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    print(f"Loading model: {model_name}")
    
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move model to device
    model = model.to(device)
    
    # Create dataset and dataloader
    print(f"Loading concept dataset from: {args.concept_dir}")
    dataset = ConceptDataset(args.concept_dir, image_processor)
    concept_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Run evaluation
    print("Running concept detection evaluation...")
    results = eval_concept_detection(model, tokenizer, concept_dataloader)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nOverall Accuracy: {results['overall_acc']:.3f}")
    print("\nPer-Concept Accuracy:")
    print("-"*30)
    
    for concept, acc in sorted(results['per_concept_acc'].items()):
        concept_name = concept.replace('_acc', '')
        print(f"{concept_name:20s}: {acc:.3f}")


if __name__ == "__main__":
    main()
