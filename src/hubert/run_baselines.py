import subprocess
import sys
import os
import argparse
import numpy as np
from loguru import logger

SEEDS = [42, 3, 1075, 495, 248, 486, 3840, 5037, 5, 59]

def _lr_full_ft(num_classes: int) -> float:
    return 0.0073 / (num_classes + 30) + 0.00002

def _max_epochs(num_classes: int) -> int:
    return 500 if num_classes >= 300 else 200

def main():
    parser = argparse.ArgumentParser(
        description='Run all 400 HuBERT baseline training runs (10 seeds x 40 class counts).'
    )
    parser.add_argument(
        '--variant', type=str, required=True, choices=['full_ft', 'ch'],
        help='full_ft: all parameters fine-tuned. ch: classifier head and projector only.'
    )
    args = parser.parse_args()

    class_counts = np.arange(25, 1001, 25)
    total = len(class_counts) * len(SEEDS)
    run = 0

    here = os.path.dirname(os.path.abspath(__file__))

    for num_classes in class_counts:
        lr = _lr_full_ft(num_classes) if args.variant == 'full_ft' else 3e-4
        epochs = _max_epochs(num_classes)
        freeze_base = 'False' if args.variant == 'full_ft' else 'True'
        freeze_feat = 'False' if args.variant == 'full_ft' else 'True'

        for seed in SEEDS:
            run += 1
            logger.info(f"[{run}/{total}] variant={args.variant} classes={num_classes} seed={seed} lr={lr:.6f} epochs={epochs}")

            cmd = [
                sys.executable, 'finetune.py',
                '--config_path', 'finetune_config.yaml',
                '--seed_everything', str(seed),
                '--max_classes', str(int(num_classes)),
                '--learning_rate', str(lr),
                '--max_epochs', str(epochs),
                '--freeze_base_model', freeze_base,
                '--freeze_feature_encoder', freeze_feat,
                '--experiment_name', args.variant,
                '--experiment_stage', 'final',
            ]

            result = subprocess.run(cmd, cwd=here)

            if result.returncode != 0:
                logger.error(f"Run failed — variant={args.variant} classes={num_classes} seed={seed}. Continuing.")

    logger.success(f"All {total} runs complete for variant={args.variant}.")

if __name__ == '__main__':
    main()
