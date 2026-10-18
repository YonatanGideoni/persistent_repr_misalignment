import argparse

from transformers import AutoImageProcessor

from coco_utils import get_conceptless_data_sim_filter, get_conceptless_data, filter_and_preprocess_datasets, \
    load_datasets
from constants import DEF_COCO_DIR, IMG_CONCEPTS_DIR
from small_scale_failures.model_utils import ADAPTERS, get_tokenizer


def train_main(args, get_config_fn, Model, train_fn, append_eos=True):
    train_ds, valid_ds, test_ds = load_datasets(data_dir=args.data_dir, coco_dataset_ratio=args.coco_dataset_ratio)

    image_processor = AutoImageProcessor.from_pretrained(args.vis_enc)
    tokenizer = get_tokenizer(args.lang_enc)

    train_ds, valid_ds, test_ds = filter_and_preprocess_datasets(
        train_ds, valid_ds, test_ds,
        image_processor, tokenizer,
        max_length=32,
        append_eos=append_eos,
    )
    if args.exclude_concepts:
        train_ds = get_conceptless_data(train_ds, tokenizer)
        valid_ds = get_conceptless_data(valid_ds, tokenizer)
        test_ds = get_conceptless_data(test_ds, tokenizer)

        if args.similarity_filter:
            train_ds = get_conceptless_data_sim_filter(train_ds)
            valid_ds = get_conceptless_data_sim_filter(valid_ds)
            test_ds = get_conceptless_data_sim_filter(test_ds)

    config = get_config_fn(args)

    model = Model(config)

    print(f'Total params: {model.num_parameters():,}')
    print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    trainer, best_checkpoint_metrics = train_fn(
        model=model,
        args=args,
        train_ds=train_ds,
        val_ds=valid_ds,
        tokenizer=tokenizer,
        image_processor=image_processor,
        concept_dir=args.concepts_dir,
    )

    print(f"Best checkpoint metrics: {best_checkpoint_metrics}")

    model = trainer.model  # Get the best model

    model.save_pretrained(args.out_dir)

    # test
    if test_ds is not None:
        test_results = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        print(f"Test results: {test_results}")


def get_base_parser(description, default_bs):
    """Arguments shared by the retrieval and generation small-scale experiments."""
    parser = argparse.ArgumentParser(description=description)

    # Data arguments
    parser.add_argument("--data_dir", type=str, default=DEF_COCO_DIR, help="Directory containing the COCO dataset")
    parser.add_argument("--coco_dataset_ratio", type=int, default=100,
                        help="Ratio of COCO dataset to use (0 to 100)")
    parser.add_argument("--exclude_concepts", action="store_true",
                        help="Whether to exclude specific concepts from the dataset")
    parser.add_argument("--similarity_filter", action="store_true",
                        help="Whether to apply similarity filtering on conceptless data")
    parser.add_argument('--concepts_dir', type=str, default=IMG_CONCEPTS_DIR,
                        help='Directory containing concept images')

    # Model arguments
    parser.add_argument("--vis_enc", type=str, default="google/vit-base-patch16-224",
                        help="Visual encoder model name/path")
    parser.add_argument("--lang_enc", type=str, default="gpt2", help="Language encoder model name/path")
    parser.add_argument("--adapter_type", type=str, default="linear", choices=list(ADAPTERS.keys()),
                        help="Type of adapter to use")

    # Training arguments
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--bs", type=int, default=default_bs, help="Batch size")
    parser.add_argument("--eval_epoch_frac", type=float, default=0.25, help="Fraction of epoch between evaluations")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloading")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Early stopping patience (in # evals, not epochs)")

    # Output arguments
    parser.add_argument("--out_dir", type=str, default="./output",
                        help="Output directory for saving model and adapters")

    return parser
