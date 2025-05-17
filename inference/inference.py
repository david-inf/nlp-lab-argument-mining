
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import plot_graph, LOG


def N(x):
    return x.detach().cpu().numpy()


def load_model(checkpoint, device):
    """Load model from hub"""
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2")
    model.to(device)
    return model, tokenizer


class InferenceDataset(Dataset):
    def __init__(self, batches, collator):
        super().__init__()
        self.batches = batches
        self.collator = collator

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx]
        return self.collator(batch)


def get_loader(dataset, tokenizer):
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        # dynamic padding, different per each batch
        padding="longest"
    )

    batches = []
    for sents, label in zip(dataset["sentences"], dataset["label"]):
        batch = []
        for sent in sents:
            encoded_sent = tokenizer(
                sent,
                max_length=128,
                truncation=True,
                return_attention_mask=True,
                return_tensors=None,
            )
            encoded_sent["labels"] = label
            batch.append(encoded_sent)
        batches.append(batch)

    new_dataset = InferenceDataset(batches, collate_fn)
    loader = DataLoader(
        new_dataset, collate_fn=collate_fn, pin_memory=True
    )

    return loader


# def compute_metrics(name, )


def inference(loader, model, device):
    """Load inference dataset and compute argumentative content metrics"""
    metrics_per_abstract = []
    # sum_of_claim_logits = 0.  # claims' logit sum for each sentence
    # sum_of_logits = 0.  # claim+premise logits for each sentence

    for doc in tqdm(loader, desc="Documents", unit="doc"):
        # for doc in tqdm(dataset, desc="Processing documents", unit="doc"):
        # 0: premise - 1: claim - 2: majclaim (abstrct+sciarg dataset)
        # class_counts = {"0": 0., "1": 0., "2": 0.}
        arg_score = 0.

        input_ids = doc["input_ids"].squeeze(0).to(device)
        attention_mask = doc["attention_mask"].squeeze(0).to(device)
        label = doc["labels"].squeeze(0)[0]  # int
        output = model(input_ids=input_ids,
                       attention_mask=attention_mask)  # logits [N, 3]

        # for sent in tqdm(doc["sentences"], desc="Processing sentences", leave=False, unit="sent"):
        # encoded = tokenizer(sent)  # tokenize sentence
        # input_ids = torch.tensor(encoded["input_ids"]).unsqueeze(0).to(device)
        # attention_mask = torch.tensor(encoded["attention_mask"]).unsqueeze(0).to(device)
        # output = model(input_ids=input_ids, attention_mask=attention_mask)  # logits

        # predicition and its logit
        # pred = np.argmax(N(output.logits), axis=1)  # int
        # sum_of_logits += np.sum(N(output.logits))  # sum logits for claims and premises
        # sum_of_claim_logits += np.max(N(output.logits)[1:])  # logit for claims
        sum_of_logits = N(output.logits).sum()  # scalar
        max_claim_logits = N(output.logits)[:, 1:].max(axis=1)
        sum_of_claim_logits = max_claim_logits.sum()

        # update class counts with current sentence class
        pred = np.argmax(N(output.logits), axis=1)  # int
        class_0 = np.where(pred == 0, 1, 0).sum()
        class_1 = np.where(pred == 1, 1, 0).sum()
        class_2 = np.where(pred == 2, 1, 0).sum()
        class_counts = {"0": class_0, "1": class_1, "2": class_2}

        # compute argumentative score
        # arg_score = sum_of_claim_logits / sum_of_logits
        # arg_score = sum_of_claim_logits / input_ids.size(0)
        # arg_score = np.max(N(output.logits)) / input_ids.size(0)
        arg_score = torch.topk(torch.from_numpy(max_claim_logits), 10).values.sum().numpy() / input_ids.size(0)
        # compute argument ratio
        arg_ratio = (class_counts["1"] + class_counts["2"])  # number of claims
        # arg_ratio = (class_counts["1"] + class_counts["2"]) / input_ids.size(0)  # fraction of claims

        # update metrics
        LOG.info("Document stats: sentences=%s, classes=%s, label=%s",
                 input_ids.size(0), class_counts, N(label))
        LOG.info("Metrics: AR=%.3f, AS=%.3f, total_logits=%.3f, claim_logits=%.3f",
                 arg_ratio, arg_score, sum_of_logits, sum_of_claim_logits)
        metrics_per_abstract.append([arg_ratio, arg_score, N(label)])

    return metrics_per_abstract  # list of list (len 2)


if __name__ == "__main__":
    device = "cuda:1"
    dataset = load_from_disk("data/inference")
    # checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_abstrct"
    # checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_sciarg"
    # checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_mixed"
    checkpoint = "david-inf/bert-sci-am"
    model, tokenizer = load_model(checkpoint, device)

    m_loader = get_loader(dataset["molecular"], tokenizer)
    m_scores = inference(m_loader, model, device)

    s_loader = get_loader(dataset["thoracic"], tokenizer)
    t_scores = inference(s_loader, model, device)

    _, axs = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
    plot_graph(np.array(m_scores), axs[0], "Molecular",
               "AR(sum of claims)", "AS(claim logits / logits)")
    plot_graph(np.array(t_scores), axs[1], "Thoracic",
               "AR(sum of claims)", "AS(claim logits / logits)")
    plt.tight_layout()

    plt.savefig("inference/plot1.svg")
