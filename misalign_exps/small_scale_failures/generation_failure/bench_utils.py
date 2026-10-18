from collections import defaultdict

import torch

from constants import device


def eval_concept_detection(model, tokenizer, concepts_dataloader):
    concepts_det = defaultdict(int)
    concepts_tot = defaultdict(int)

    with torch.no_grad():
        model.eval()
        for batch in concepts_dataloader:
            pixel_values = batch['pixel_values'].to(device)
            concepts = batch['concept']

            outputs = model.generate(pixel_values=pixel_values, do_sample=False)

            captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for caption, concept in zip(captions, concepts):
                concepts_tot[concept] += 1
                if concept in caption.lower():
                    concepts_det[concept] += 1

    concepts_acc = {f'{c}_acc': concepts_det[c] / concepts_tot[c] for c in concepts_tot.keys()}
    overall_accuracy = sum(concepts_det.values()) / sum(concepts_tot.values())

    return {
        'per_concept_acc': concepts_acc,
        'overall_acc': overall_accuracy,
    }
