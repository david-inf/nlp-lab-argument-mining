"""BERT family models for sequence classification"""

import os
from types import SimpleNamespace

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model


def get_bert(opts):
    """
    Get BERT pretrained model and its tokenizer for finetuning
    - DistilBERT (uncased)
    - SciBERT (uncased)
    - SentenceBERT (MiniLM-L6)
    """
    if opts.model == "distilbert":
        checkpoint = "distilbert-base-uncased"
    elif opts.model == "scibert":
        checkpoint = "allenai/scibert_scivocab_uncased"
    elif opts.model == "sbert":
        checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        raise ValueError(f"Unknown BERT model {opts.model}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2)
    
    tokenizer, model = _finetuning_setting(opts, tokenizer, model)
    return tokenizer, model


# TODO: single function for inference?
def get_finetuned_bert(opts):
    """Load finetuned BERT model"""
    # check if directory where pretrained model is saved exists
    if not os.path.exists(opts.checkpoint):
        raise ValueError(f"Pretrained model directory {opts.checkpoint} does not exist")

    tokenizer = AutoTokenizer.from_pretrained(opts.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        opts.checkpoint, num_labels=2)

    return tokenizer, model


def _finetuning_setting(opts, tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification):
    """Set BERT model finetuning settings"""
    ft_setting = SimpleNamespace(**opts.ft_setting)

    if ft_setting.type == "full":
        # train all parameters
        pass

    elif ft_setting.type == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=ft_setting.rank, lora_alpha=ft_setting.alpha,
            lora_dropout=0.0, bias="none",
            target_modules=ft_setting.target_modules
        )
        model = get_peft_model(model, peft_config)

    else:
        raise ValueError(f"Unknown training setting {opts.ft_setting}")

    return tokenizer, model
