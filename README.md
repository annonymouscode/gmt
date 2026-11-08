# Gated Marginal Transport (GMT)

Code for the anonymous CVPR 2027 submission *When All Classes Compete: Base-Class Bias in Prompt-Tuned CLIP and a Gated, Label-Free Correction*.

GMT corrects the predictions of a prompt-tuned CLIP model when base and novel classes compete.
It projects a batch of predictions onto a uniform class marginal with Sinkhorn iterations and applies the projection through a gate read from the class balance of frozen zero-shot predictions.
The gate is calibrated on a pseudo base/novel split of the base training data, without novel classes or test labels.

## Layout
- `trustprompt/`: datasets, CLIP prompt learners (CoOp, CoCoOp, KgCoOp, IVLP, MaPLe, PromptSRC) and training engine.
- `run_textprompt.py`, `dump_logits.py`: train a learner and save test logits (adapted and zero-shot).
- `correct_bias.py`, `adaptive_correct.py`: Sinkhorn projection, Gini gate and GMT.
- `calibrate_mixed.py`, `run_calib.sh`: label-free calibration of the gate on pseudo-splits.
- `baselines_cs.py`, `extra_baselines.py`, `ablate_gate.py`, `analysis_suite.py`, `supp_analyses.py`, `batch_test.py`: baselines, ablations and analyses.
- `newdata_analysis.py`, `run_newdata.sh`: datasets never used for calibration.
- `make_assets.py`: builds every table, figure and in-text number of the paper from `runs/`.

## Setup
Python >= 3.9, PyTorch, and OpenAI CLIP (`pip install git+https://github.com/openai/CLIP.git`), plus numpy, scipy, matplotlib and POT (`pip install pot`).
Datasets follow the CoOp layout and base/novel splits; see `scripts/download_data.sh` and check them with `python scripts/check_data.py`.

## Reproducing the paper
1. Train each learner and save logits: `run_tp.sh` / `run_learner.sh` (arguments in the script headers), or `run_seed.sh` for all seeds.
2. Calibrate the gate on pseudo-splits: `run_calib.sh`.
3. Corrections, baselines and ablations: `post_learner.sh`, then `python extra_baselines.py`.
4. New datasets: `run_newdata.sh`, then `python newdata_analysis.py`.
5. Tables, figures and numbers: `python make_assets.py` (writes `paper_assets/`).

Every number in the paper is produced by `make_assets.py`; nothing is typed by hand.
