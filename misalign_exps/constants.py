import os

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# HF datasets cache dir for COCO 2014, override with the COCO_DIR env var
DEF_COCO_DIR = os.environ.get('COCO_DIR', './coco2014')
# Held-out concept test set, rebuilt from ImageNet by ../concept_testset/rebuild.py; override with the IMG_CONCEPTS_DIR env var
IMG_CONCEPTS_DIR = os.environ.get('IMG_CONCEPTS_DIR', '../concept_testset/img_concepts_outs')

DEF_CONCEPTS = ('zebra', 'banana', 'umbrella', 'toothbrush', 'pizza')

def get_concepts():
    concepts = os.environ.get('CONCEPTS', None)
    if concepts is None:
        print('No concepts given from env variables, using default concepts')
        return DEF_CONCEPTS

    concepts = concepts.split(',')
    print(f'Using concepts: {concepts}')

    return concepts
