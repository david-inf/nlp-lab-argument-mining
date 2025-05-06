# BERT for scientific literature argument mining


## Phase 1: Finetuning BERT for argument component detection

### :one: Baseline

- `python src/baseline.py --dataset "abstrct" --extractor "distilbert" --classifier "svm"`

| Extractor for LinearSVC   | size | Dataset | `train_acc` | `val_acc` |
| ------------------------- | ---- | ------- | ----------- | --------- |
| `distilbert-base-uncased` | 67M  | AbstRCT | 0.849       | 0.822     |

- `python src/baseline.py --dataset "abstrct" --extractor "distilbert" --classifier "logistic"`

| Extractor for LogisticRegression | size | Dataset | `train_acc` | `val_acc` |
| -------------------------------- | ---- | ------- | ----------- | --------- |
| `distilbert-base-uncased`        | 67M  | AbstRCT | 0.840       | 0.822     |


## Phase 2: Deploy BERT on medical literature
