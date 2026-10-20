# Code for "Persistent Representational Misalignment Induces Vision-Language Model Failures"

[![arXiv](https://img.shields.io/badge/arXiv-2610.25252-b31b1b.svg)](https://arxiv.org/abs/2610.25252)

This is the code release for [Persistent Representational Misalignment Induces Vision-Language Model Failures](https://arxiv.org/abs/2610.25252). 

## Setup

This repo requires two venvs, one for the LLaVA experiments and one for everything else. To set up each one:
```
# main venv (everything outside llava/), creates the `misalign_exps` conda env
bash misalign_exps/setup_env.sh
# llava venv, creates the `ktt_llava` conda env and the patched lmms-eval/ and bongard/ clones in llava/
bash llava/setup_env.sh
```

For credentials (e.g. accessing HuggingFace datasets), please copy .env.example to .env and fill it in.

Experiments require the following datasets, where depending which you run you can likely download only a subset thereof.
1. ImageNet, for making the held-out concepts dataset and the small-scale validation metrics.
2. The LLaVA training dataset, see [LLaVA's data instructions](llava/docs/Data.md) for details.
3. VisLingA, for the vision/language analogies test. This dataset is on HF so no special download is required a priori, see [VisLingA on HF](https://huggingface.co/datasets/YGideoni/VisLingA).
4. The COCO dataset. Note that the full processed COCO dataset comes out to ~340GB, so you may want to process a subset - see `load_datasets` in `misalign_exps/coco_utils.py` (set via the training scripts' `--coco_dataset_ratio` arg).
5. For evaluating the VLM failures, Bongard in Wonderland needs to be downloaded from [Bongard in Wonderland](https://github.com/ml-research/bongard-in-wonderland) ("Download the data") into `llava/bongard/data/bpimgs`.
The remaining benchmarks are downloaded automatically from HF. Some datasets and models are gated, so if using them set `HF_TOKEN` in `.env`.

## Running the experiments

### Broken conditions for representational alignment

The analogies experiments using VisLingA are in `misalign_exps/vislinga/`, to test how well a given model's embeddings recognises the analogies run:
```
# for a text model:
python misalign_exps/vislinga/eval_vislinga.py --modality text --model all-MiniLM-L12-v2 --out misalign_exps/vislinga/results
# for a vision model:
python misalign_exps/vislinga/eval_vislinga.py --modality image --model facebook/dinov3-vitb16-pretrain-lvd1689m --out misalign_exps/vislinga/results
```
`misalign_exps/vislinga/vislinga_sweep.job` runs all the models from the paper, then `python misalign_exps/vislinga/collect_results.py --results misalign_exps/vislinga/results` summarises the results.

For the cosine similarity RMSE experiments, see `misalign_exps/similarity_structure/cossim_rmse.py`. To reproduce them, run:
```
cd misalign_exps && python -m similarity_structure.cossim_rmse
```

### Small-scale generalization failures
To construct the held-out concepts dataset, please run:
```
python concept_testset/rebuild.py --imagenet_dir /path/to/imagenet
```

To train models with/without the held-out concepts:
```
cd misalign_exps
# generation, with / without the held-out concepts
sbatch scripts/gen/linear.job
sbatch scripts/gen/linear_nocsf.job
# retrieval, with / without the held-out concepts
sbatch scripts/clip/clip_training_scripts/linear_regular_cls.sh
sbatch scripts/clip/clip_training_scripts/linear_nocsf_cls.sh
```
Edit the `/path/to/...` placeholders at the top of the scripts before running.
Confidence intervals over the per-concept accuracies are computed with `python bootstrap_ci.py --accs ...`.

### LLaVA experiments

The LLaVA code here is a modified version of the [LLaVA repo](https://github.com/haotian-liu/LLaVA). 

The LLaVA jobs read the data, checkpoint and held-out concepts dataset locations from `DATA_DIR`, `CKPT_DIR` and `CONCEPT_TESTSET_DIR` (set at the top of each job script or the `.sh` script it calls, or exported before submitting).

To train LLaVA models with/without the held-out concepts (submitted from `llava/`):
```
cd llava
# without the held-out concepts, for each of the three concept sets A, B, C
sbatch slurm_scripts/vit/pretrain_heldout_concA.job   # then finetune_heldout_concA.job; same for concB, concC
# with the full dataset
sbatch slurm_scripts/vit/pretrain_full_data.job       # then finetune_full_data.job
# concept detection for a trained model
python bench_llava.py --model_paths $CKPT_DIR/llava-v1.5-7b-lora-vit
```

And to train LLaVA models with the various ViT backbones:
```
cd llava
# the 11 base backbones: slurm_scripts/b/<backbone>/{pret,ft_lora}.job
sbatch slurm_scripts/b/vitb/pret.job                  # then slurm_scripts/b/vitb/ft_lora.job
# the 26 finetuned backbones
python slurm_scripts/vit_finetunes/gen_finetune_jobs.py
bash slurm_scripts/vit_finetunes/submit_all.sh
```
These models can be evaluated on various benchmarks exhibiting VLM failures by running:
```
cd llava
python bench_llava_wino.py --ckpt_dir $CKPT_DIR                        # Winoground
sbatch slurm_scripts/eval/vlmbias/eval_all.job                         # VLMBias (+ eval_all_finetunes.job)
sbatch slurm_scripts/eval/vlms_are_blind/eval_parallel.job             # VLMs Are Blind (+ eval_finetunes.job)
sbatch slurm_scripts/eval/mmvp/eval_all.job                            # MMVP
sbatch slurm_scripts/eval/omnispatial/eval_all.job                     # OmniSpatial
sbatch slurm_scripts/eval/bongard/eval_all_models.job                  # Bongard
```
The backbones' representational alignment scores are computed with `cd misalign_exps/repr_alignment && python compute_alignment.py`.

## Licenses

Everything outside `llava/` is under the MIT license (`LICENSE`). `llava/` is a fork of the original LLaVA repo and has its Apache-2.0 license (`llava/LICENSE`). ImageNet images remain under ImageNet's terms of use.

## Bibtex

```bibtex
@article{gideoni2026persistent,
  title={Persistent Representational Misalignment Induces Vision-Language Model Failures},
  author={Gideoni, Yonatan and Gelberg, Yoav and Rudner, Tim G. J. and Gal, Yarin},
  journal={arXiv preprint arXiv:2610.25252},
  year={2026}
}
```
