# BERT for scientific literature Argument Mining

The idea is to deploy a finetuned BERT on scientific literature in order to gain information about the argumentative content of a scientific articles from medical literature.

<details>
<summary>Code organization</summary>

- `data/`
  - `finetuning/` AbstRCT and SciArg datasets, and a merged version of these that will be the actual training dataset
  - `inference/` two sets of medical literature abstracts
- `inference/`
  - `inference.py` inference main program on articles from medical literature
  - `load_and_eval.py` load validation dataset from training and compute validation accuracy
  - `utils.py` inference utilities
- `src/`
  - `ckps/`
  - `configs/` configuration files
    - Contains `generate_config.py` for automatic configuration files generation
  - `models/` model definitions
  - `results/`
  - `utils/` various utilities in `misc_utils.py` and `train_utils.py`
  - `baseline.py` baseline with base machine learning models to improve
  - `cmd_args.py` main programs arguments
    - `python src/main.py --help`
    - `python src/ftdata.py --help`
  - `ftdata.py` utilities for loading datasets
  - `main.py` main program for finetuning BERT family models
  - `train.py` training loop

</details>


## Phase 1: Finetuning BERT for argument component detection

Here we search for the best finetuned model on the argument component detection task, that is a sentence classification task.

### :zero: Dataset & Models

The finetuning dataset is a merged version of the two following datasets

- [pie/abstrct](https://huggingface.co/datasets/pie/abstrct)
- [pie/sciarg](https://huggingface.co/datasets/pie/sciarg)

Follows models under finetuning:

- [distilbert](https://huggingface.co/distilbert/distilbert-base-uncased)
- [sbert](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)


### :one: Baseline

Use a feature extraction pipeline to obtain a baseline to improve with finetuning. That is the two models that will be finetuned are here used as feature extractors, on top of their representation, we train a logistic regression as a baseline.

- `python src/baseline.py --dataset "abstrct" --extractor "distilbert"`

| Extractor | size  | Dataset | `train_acc` | `val_acc` |
| -------------------------------- | ----- | ------- | ----------- | --------- |
| `distilbert`        | 67M   | `mixed` | 0.840       | 0.822     |
| `sbert`               | 22.7M | `mixed` | 0.          | 0.        |


### :two: Finetuning

- `python src/main.py --config src/configs/mixed/distilbert_full_mixed.yaml`
- `python src/main.py --config src/configs/mixed/sbert_full_mixed.yaml`

| model        | setting           | val_acc |
| ------------ | ----------------- | ------- |
| `distilbert` | `full-finetuning` | 0.      |
| `sbert`      | `full-finetuning` | 0.      |


## Phase 2: Deploy BERT on medical literature

We deploy the finetuned BERT on two datasets containing medical literature abstracts

For example, do inference on cuda with metrics pair 1

```bash
python inference/inference.py --device cuda --metric_id 1
```

<p align="middle">
  <img src="inference/results/plot_0.svg", alt="metrics pair 0" width="30%">
  &nbsp;
  <img src="inference/results/plot_1.svg", alt="metrics pair 1" width="30%">
  &nbsp;
  <img src="inference/results/plot_2.svg", alt="metrics pair 2" width="30%">
</p>

<p align="middle">
  <img src="inference/results/plot_3.svg", alt="metrics pair 3" width="30%">
  &nbsp;
  <img src="inference/results/plot_4.svg", alt="metrics pair 4" width="30%">
</p>
