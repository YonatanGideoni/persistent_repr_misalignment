from torch.utils.data import DataLoader
import evaluate
import numpy as np
from transformers import EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer

from small_scale_failures.clip_failure.bench_utils import ConceptDataset
from small_scale_failures.generation_failure.bench_utils import eval_concept_detection


def train_gen(model, args, train_ds, val_ds, tokenizer, image_processor, concept_dir):
    # eval approx every x epochs
    eval_steps = int(args.eval_epoch_frac * len(train_ds) / args.bs)
    warmup_steps = int(0.05 * len(train_ds) / args.bs)

    dataset = ConceptDataset(concept_dir, image_processor)
    concept_dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

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
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=500,
        save_steps=eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="wandb",
        dataloader_num_workers=args.num_workers,
    )

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        # hf trainer bug https://github.com/huggingface/transformers/issues/22634
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        rouge_result = rouge.compute(predictions=pred_str, references=labels_str)
        rouge_result = {k: v * 100 for k, v in rouge_result.items()}

        checkable_preds, checkable_labels = zip(*[(p, l) for p, l in zip(pred_str, labels_str) if len(p) > 1])
        bleu_result = bleu.compute(predictions=checkable_preds, references=checkable_labels)

        concept_det_res = eval_concept_detection(model, tokenizer, concept_dataloader)

        return {
            **rouge_result,
            **concept_det_res['per_concept_acc'],
            'overall_concept_acc': concept_det_res['overall_acc'],
            "bleu": bleu_result["bleu"] * 100,
            "gen_len": bleu_result["translation_length"] / len(preds)
        }

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    return trainer, trainer.state.best_metric
