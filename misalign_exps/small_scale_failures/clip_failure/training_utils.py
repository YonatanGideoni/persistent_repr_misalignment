from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet
from transformers import Trainer, Seq2SeqTrainingArguments, EarlyStoppingCallback

from small_scale_failures.clip_failure.bench_utils import setup_concept_detection_data, eval_concept_detection, get_imagenet_class_names, \
    compute_imagenet_accuracy


def setup_imagenet_evaluation(tokenizer, args):
    """Set up ImageNet validation dataset and tokenized prompts."""
    # Prepare ImageNet validation dataset
    imagenet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imagenet_val = ImageNet(
        root=args.imagenet_dir,
        split='val',
        transform=imagenet_transform
    )

    imagenet_loader = DataLoader(
        imagenet_val,
        batch_size=args.bs * 4,  # Can use larger batch for inference
        shuffle=False,
        num_workers=args.num_workers
    )

    # Get class names and create text prompts
    imagenet_classes = get_imagenet_class_names()
    text_prompts = [f"a photo of a {class_name}" for class_name in imagenet_classes]

    # Tokenize all prompts
    encoded_prompts = tokenizer(
        text_prompts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=77
    )

    return imagenet_loader, encoded_prompts


def train_clip(model, args, train_ds, val_ds, tokenizer, image_processor, concept_dir):
    # Set up ImageNet evaluation
    imagenet_loader, encoded_prompts = setup_imagenet_evaluation(tokenizer, args)
    # set up held out concept evaluation
    concepts_dataloader, all_prompts, concept_indices = setup_concept_detection_data(
        concept_dir, image_processor, batch_size=args.bs
    )

    # eval approx every x epochs
    eval_steps = int(args.eval_epoch_frac * len(train_ds) / args.bs)
    warmup_steps = int(0.05 * len(train_ds) / args.bs)

    # Setup training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.n_epochs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=500,
        save_steps=eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="imagenet_top1",
        greater_is_better=True,
        report_to="wandb",
        dataloader_num_workers=args.num_workers,
    )

    temperature = model.config.temperature

    def compute_metrics(eval_preds):
        logits = eval_preds.predictions
        clip_score = logits.diagonal().mean() * temperature

        # Compute ImageNet zero-shot accuracy
        imagenet_acc = compute_imagenet_accuracy(
            model, imagenet_loader, encoded_prompts,
        )

        # Compute concept detection accuracy
        concept_det_res = eval_concept_detection(model, tokenizer, all_prompts, concept_indices, concepts_dataloader)

        return {
            "clip_score": clip_score.item(),
            "imagenet_top1": imagenet_acc,
            'per_concept_accuracy': concept_det_res['per_concept_accuracy'],
            'conc_overall_accuracy_top1': concept_det_res['overall_accuracy_top1'],
            'conc_overall_accuracy_top5': concept_det_res['overall_accuracy_top5'],
        }

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer, trainer.state.best_metric
