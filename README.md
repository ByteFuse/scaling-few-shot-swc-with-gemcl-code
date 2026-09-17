# Scaling Few-Shot Spoken Word Classification with Generative Meta-Continual Learning

Code for the Interspeech 2026 paper *Scaling Few-Shot Spoken Word Classification with Generative Meta-Continual Learning*.

---

## Repo structure

```
recordings_used/          # the episode recordings from our paper's GeMCL run, kept for reference
  episode_0/
    train_links.csv
    test_links.csv
  episode_1/ ... episode_9/

meta_splits/              # meta-train and meta-test word class splits used in the paper
  train_5_shots_test_5_shots/
    en/
      meta_train_classes.txt   # words used for meta-training (8915 words)
      meta_test_classes.txt    # words used for meta-testing (3821 words)
      meta_data_classes.txt    # split metadata: random seed, total/train/test class counts

src/
  gemcl/                  # GeMCL model and training/testing scripts
    model/
      gemcl.py
      encoder.py
    meta_dataset.py
    utils.py
    single_lang_meta_train.py
    single_lang_meta_test.py
  hubert/                 # HuBERT baseline fine-tuning scripts
    finetune.py
    finetune_config.yaml
    data_module_with_val.py
    hubert_pretrain_model.py
    processing_utils.py
```

---

## Data setup

Download the English portion of the [Multilingual Spoken Words Corpus (MSWC)](https://mlcommons.org/datasets/multilingual-spoken-words-corpus/) and organise it as follows relative to the repo root:

```
data/
  macroset/
    en/
      clips/
        {word}/
          {recording}.opus

splits/
  en/
    en_train.csv
    en_test.csv
```

Both `clips/` and the validity CSVs come from the MSWC download. The `meta_splits/` directory is already included in this repo.

---

## Reproducing the experiments

The pipeline runs in order: **GeMCL meta-train → GeMCL meta-test → HuBERT baselines**.

GeMCL meta-testing randomly samples 10 episodes and saves the recordings it used. The HuBERT baseline scripts then read from those saved episode directories — this is how both models end up evaluated on the exact same recordings.

The `recordings_used/` directory in this repo contains the episodes from our paper's GeMCL run. Running GeMCL yourself will produce different random episodes.

---

## Running GeMCL

All GeMCL commands must be run from `src/gemcl/`. Checkpoints and experiment outputs are written relative to that directory. Data and split paths are resolved relative to the repo root.

### Meta-training

```bash
python single_lang_meta_train.py --language en --seed 42
```

Checkpoints are saved to `./checkpoints/meta_train_wav2vec2/25way_5shot/`.

### Meta-testing

```bash
python single_lang_meta_test.py --seed 42
```

The test script samples 10 episodes from the meta-test word pool, runs the continual learning evaluation, and saves the episode recordings to:

```
src/gemcl/experiments/meta-test/en_1000-way-5-shot/
  episode_0/
    train_links.csv
    test_links.csv
  episode_1/ ... episode_9/
```

These are the directories the HuBERT scripts will read from.

---

## Running HuBERT baselines

All HuBERT commands must be run from `src/hubert/`. Run GeMCL meta-testing first so the episode directories exist.

Two variants were evaluated:

- **Full FT** — all HuBERT parameters fine-tuned
- **CH** — backbone frozen, classifier head and projector only

To reproduce all 400 runs per variant (10 seeds × 40 class counts):

```bash
python run_baselines.py --variant full_ft
python run_baselines.py --variant ch
```

`run_baselines.py` handles the learning rate schedule (Full FT uses a per-class-count formula; CH uses a fixed `3e-4`), the epoch schedule (200 epochs for <300 classes, 500 for ≥300), and the correct freeze flags per variant. Results are saved under `classification/experiment_{variant}/final/`.

To run a single experiment manually, use `finetune.py` directly — see the arguments at the top of that file and `finetune_config.yaml` for defaults.

Test results are saved to `classification/experiment_{name}/{stage}/{experiment_id}/best_model/final_test_results.csv`.

---

## Logging

Training metrics are logged to the terminal. You can plug in your own logger (e.g. W&B, TensorBoard) if you want experiment tracking.
