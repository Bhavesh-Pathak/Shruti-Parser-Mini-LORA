import time
import json
import math
import os
import argparse
import random

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score


def measure_latency_and_throughput(model, tokenizer, texts, device, repeat=20):
    model.eval()
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
    # warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(**inputs)
    t_total = time.time() - t0
    avg_latency = t_total / repeat
    batch_token_count = sum([len(tokenizer(tok)['input_ids']) for tok in texts])
    tokens_per_sec = batch_token_count / avg_latency
    return avg_latency, tokens_per_sec


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(pred.label_ids, preds)
    f1 = f1_score(pred.label_ids, preds, zero_division=0)
    return {"eval_accuracy": acc, "eval_f1": f1}


def get_preds(model, dataset, device):
    model.eval()
    preds = []
    labels = []
    for i in range(0, len(dataset)):
        item = {k: torch.tensor([v]).to(device) for k, v in dataset[i].items() if k in ['input_ids','attention_mask']}
        with torch.no_grad():
            out = model(**item)
            logits = out.logits.cpu().numpy()[0]
            preds.append(int(np.argmax(logits)))
            labels.append(int(dataset[i]['labels']))
    return preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/mini_dataset.csv')
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df['label'] = df['label'].astype(int)

    # small split
    train_df = df.sample(frac=0.83, random_state=42)
    test_df = df.drop(train_df.index)
    train = Dataset.from_pandas(train_df.reset_index(drop=True))
    test = Dataset.from_pandas(test_df.reset_index(drop=True))

    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def preprocess(batch):
        toks = tokenizer(batch['text'], truncation=True, padding=False)
        toks['labels'] = batch['label']
        return toks

    train = train.map(preprocess, batched=True)
    test = test.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir='./outputs',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy='no',
        disable_tqdm=True,
        fp16=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model.to(device)

    # evaluate base model
    trainer_base = Trainer(
        model=base_model,
        args=training_args,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    eval_base = trainer_base.evaluate()

    sample_texts = test['text'] if len(test) > 0 else train['text'][:2]
    # ensure python str objects (datasets may return numpy/pandas types)
    sample_texts = [str(t) for t in sample_texts]
    base_latency, base_tps = measure_latency_and_throughput(base_model, tokenizer, sample_texts, device)

    # LoRA model
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    # DistilBERT attention uses q_lin/k_lin/v_lin naming for projection layers
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "k_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    peft_model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_lora = trainer.evaluate()

    peft_model.to(device)
    lora_latency, lora_tps = measure_latency_and_throughput(peft_model, tokenizer, sample_texts, device)

    base_preds, base_labels = get_preds(base_model, test, device)
    lora_preds, lora_labels = get_preds(peft_model, test, device)

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        'base_eval': eval_base,
        'lora_eval': eval_lora,
        'base_latency_s': base_latency,
        'base_tokens_per_sec': base_tps,
        'lora_latency_s': lora_latency,
        'lora_tokens_per_sec': lora_tps,
        'test_size': len(test),
        'base_preds': base_preds,
        'lora_preds': lora_preds,
        'labels': base_labels,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    pd.DataFrame([{
        'model': 'base',
        'accuracy': eval_base.get('eval_accuracy', None),
        'f1': eval_base.get('eval_f1', None),
        'latency_s': base_latency,
        'tokens_per_sec': base_tps
    }, {
        'model': 'lora',
        'accuracy': eval_lora.get('eval_accuracy', None),
        'f1': eval_lora.get('eval_f1', None),
        'latency_s': lora_latency,
        'tokens_per_sec': lora_tps
    }]).to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)

    print(f"Wrote results to {args.output_dir}/results.json and results.csv")


if __name__ == '__main__':
    main()
