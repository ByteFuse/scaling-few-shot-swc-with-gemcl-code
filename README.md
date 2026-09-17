# Scaling Few-Shot Spoken Word Classification with Generative Meta-Continual Learning

Code for the Interspeech 2026 paper *Scaling Few-Shot Spoken Word Classification with Generative Meta-Continual Learning*.

---

## Repo structure

```
recordings_used/
  episode_0/
    train_links.csv
    test_links.csv
  episode_1/ ... episode_9/

meta_splits/
  train_5_shots_test_5_shots/
    en/
      meta_train_classes.txt
      meta_test_classes.txt
      meta_data_classes.txt

src/
  gemcl/
    model/
      gemcl.py
      encoder.py
    meta_dataset.py
    utils.py
    single_lang_meta_train.py
    single_lang_meta_test.py
  hubert/
    finetune.py
    finetune_config.yaml
    run_baselines.py
    data_module_with_val.py
    hubert_pretrain_model.py
    processing_utils.py
```

### File descriptions

| File | Description |
|---|---|
| `src/gemcl/model/gemcl.py` | GeMCL model: Normal-Gamma generative classifier with meta-training forward pass and continual learning inference (`learn_class_statistics`, `test_forward`) |
| `src/gemcl/model/encoder.py` | Wav2Vec2-inspired transformer encoder (12 layers, 12 heads) that takes MFCCs as input and produces embeddings |
| `src/gemcl/meta_dataset.py` | `IterableDataset` that streams N-way K-shot episodes sampled from MSWC audio files |
| `src/gemcl/utils.py` | MFCC preparation, meta-split generation, episode file writing, and timing/reporting utilities |
| `src/gemcl/single_lang_meta_train.py` | Meta-trains GeMCL on 25-way-5-shot episodes for 5000 steps |
| `src/gemcl/single_lang_meta_test.py` | Continual learning evaluation: sequentially learns 25→1000 classes across 10 episodes and records per-class accuracy at each step |
| `src/hubert/finetune.py` | Fine-tunes HuBERT for a single (seed, num_classes) combination; saves the best checkpoint and per-word test results |
| `src/hubert/run_baselines.py` | Orchestrates all 400 baseline runs (10 seeds × 40 class counts) for a given variant (`full_ft` or `ch`) |
| `src/hubert/data_module_with_val.py` | PyTorch Lightning `DataModule`: loads MSWC audio, applies k-shot sampling, and scrounges a validation set from unused training recordings |
| `src/hubert/hubert_pretrain_model.py` | PyTorch Lightning `LightningModule` wrapping HuBERT for sequence classification, with accuracy and F1 logging |
| `src/hubert/processing_utils.py` | Utilities for reading episode CSVs and selecting the first N words from a training episode |
| `src/hubert/finetune_config.yaml` | Default hyperparameters for `finetune.py` |
| `meta_splits/.../meta_train_classes.txt` | The 8915 MSWC words reserved for meta-training |
| `meta_splits/.../meta_test_classes.txt` | The 3821 MSWC words reserved for meta-testing |
| `meta_splits/.../meta_data_classes.txt` | Split metadata: random seed used and class counts |
| `recordings_used/episode_N/{train,test}_links.csv` | The exact MSWC recordings used in our paper's GeMCL meta-test run, kept for reference |

---

## Data setup

Download the English portion of the [Multilingual Spoken Words Corpus (MSWC)](https://huggingface.co/datasets/MLCommons/ml_spoken_words) and organise it as follows relative to the repo root:

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

---

## Citation

```bibtex
@inproceedings{beyers2026scaling,
  title={Scaling few-shot spoken classification with generative meta-continual learning},
  author={Beyers, Louise and Ziki, Batsirayi Mupamhi and van der Merwe, Ruan},
  booktitle={Proceedings of Interspeech 2026},
  year={2026}
}
```
