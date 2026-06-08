#!/usr/bin/env python3
"""
ACC LLM Quick Reference — Print common commands and file locations.

Usage: python scripts/quick_reference.py
"""

REF = """
================================================================================
                    ACC LLM — QUICK REFERENCE
================================================================================

ENVIRONMENT
-----------
  Check environment readiness:
    python scripts/check_environment.py

  Check GPU:
    python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

MODEL DOWNLOAD
--------------
  Authenticate HF Hub (required):
    huggingface-cli login

  Download Mistral 7B:
    python scripts/download_model_robust.py

  Verify download:
    python scripts/validate_model_load.py

DATASETS
--------
  Auto-load all available datasets:
    python experiments/datasets/auto_load.py

  Generate calibration prompts:
    python scripts/generate_calibration_prompts.py \
        --dataset experiments/datasets/pubmedqa/train.jsonl \
        --output data/calibration/pubmedqa_cal.jsonl \
        --num-prompts 100

TRAINING
--------
  QLoRA fine-tuning (desktop):
    python scripts/train.py --config configs/desktop_qlora.yaml

  QLoRA fine-tuning (Jetson):
    python scripts/train.py --config configs/jetson_qlora.yaml

  Auto-launcher (waits for model, then trains):
    python scripts/auto_launch_training.py

ACC LAYER TRAINING
------------------
  Generate conflict detector training data:
    python scripts/generate_conflict_data.py \
        --model models/mistral_7b \
        --adapter adapters/desktop_run/final_adapter \
        --output data/acc_training/mistral_conflict_data.jsonl

  Train conflict detector:
    python scripts/train_conflict_detector.py \
        --data data/acc_training/mistral_conflict_data.jsonl \
        --save_dir adapters/acc_conflict_detector

EVALUATION
----------
  Run ACC validation:
    python scripts/validate_acc.py \
        --adapter adapters/desktop_run/final_adapter \
        --config configs/desktop_qlora.yaml \
        --dataset experiments/datasets/pubmedqa/test.jsonl

  Run ablation experiments:
    python experiments/run_ablation.py --config configs/desktop_qlora.yaml

  Aggregate results:
    python scripts/aggregate_results.py --input results/ --output results/summary/

INFERENCE
---------
  Run inference with adapter:
    python scripts/infer.py \
        --model models/mistral_7b \
        --adapter adapters/desktop_run/final_adapter \
        --prompt "Your prompt here"

KEY FILES
---------
  Paper draft:              paper/draft.md
  Enhanced related work:    paper/related_work_enhanced.md
  References (BibTeX):      paper/references.bib
  Experimental protocol:    experiments/EXPERIMENTAL_PROTOCOL.md
  Wake-up guide:            results/WAKE_UP_GUIDE.md
  Status report:            results/overnight_status_report.md
  Changelog:                results/OVERNIGHT_CHANGELOG.md

CONFIGS
-------
  Desktop (RTX 3080):       configs/desktop_qlora.yaml
  Jetson (Orin Nano):       configs/jetson_qlora.yaml
  Test (tiny GPT-2):        configs/acc_test.yaml

TROUBLESHOOTING
---------------
  CUDA OOM during training:
    - Reduce max_seq_length to 512
    - Reduce per_device_train_batch_size to 1
    - Increase gradient_accumulation_steps to 8

  Model load fails:
    - Re-run download script
    - Check scripts/validate_model_load.py output

  HF Hub rate limited:
    - huggingface-cli login
    - Or: HF_TOKEN=xxx python scripts/download_model_robust.py

  WandB not needed:
    - Training works without it (report_to=[])

================================================================================
"""

print(REF)
