# -*- coding: utf-8 -*-
"""
Created on Wed May 28 01:59:24 2025

@author: harun
"""


from comet_ml import start
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, EvalPrediction
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss, confusion_matrix, classification_report
from scipy.special import expit
import evaluate
import numpy as np
import ipdb
import pandas as pd
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")
device=accelerator.device

model_id = "USERNAME/MODEL_LINK_ON_HUGGINGFACE" 
revision_id = "THE_SPECIFIC_REVISION_ID_TO_USE" # "main"
dataset_id = "USERNAME/PREPROCESSED_DATASET_ON_HUGGINGFACE"

label2id = {'LIN': 0, 'SI': 1, 'CL': 2, 'D': 3, 'HI': 4, 'PL': 5, 'TI': 6, 'PC': 7}
id2label = {0: 'LIN', 1: 'SI', 2: 'CL', 3: 'D', 4: 'HI', 5: 'PL', 6: 'TI', 7: 'PC'}
num_labels = 8

model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision_id, reference_compile=False)
model.to(device)

dataset = load_dataset(dataset_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(data):
    return tokenizer(data["text"], truncation=False)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # first calculate the predictions given a threshold
    threshold = 0.5

    predictions = (expit(logits) > threshold).astype(int)
    
    return_dict = {}
    f1_scores = []
    hamming_scores = []
    
    # for each class, do the metric calculation
    for i in range(num_labels):
        class_name = "classification_report_" + id2label[i]
        ham_name = "hamming_" + id2label[i]
        
        return_dict[class_name] = classification_report(labels[:, i], predictions[:, i], target_names=["~"+id2label[i], id2label[i]], output_dict=True)
        return_dict[ham_name] = hamming_loss(labels[:, i], predictions[:, i])
        
        f1_scores.append(return_dict[class_name]['macro avg']["f1-score"])
        hamming_scores.append(return_dict[ham_name])
    
    return_dict["global_avg"] = {"f1-score" : np.mean(f1_scores), "hamming" : np.mean(hamming_scores)}
    
    return return_dict

batch_size = 8

training_args = TrainingArguments(
    report_to="none",
    per_device_eval_batch_size=batch_size,
    bf16=True
  )

trainer = accelerator.prepare(Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator, 
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    ))


logits, labels, metrics = trainer.predict(tokenized_datasets["test"])
par_ids = tokenized_datasets["test"]["par_ids"]
text = dataset["test"]["text"]


threshold = 0.5
predictions = (expit(logits) >= threshold).astype(int)
out_predictions = [np.array2string(predictions[i], separator='')[1:-1] for i in range(len(predictions))]

out_labels = np.copy(labels.astype(int))
out_labels = [np.array2string(out_labels[i], separator='')[1:-1] for i in range(len(out_labels))]

df = {"ID" : list(par_ids),
      "Text" : list(text),
      "Logits" : list(logits),
      "Predicted Labels" : list(out_predictions),
      "Gold Labels" : list(out_labels)}

df = pd.DataFrame(df)
df.to_csv("Inference_Results.csv", index=True)


with open('metrics.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")
    

# all zeros
####################
out_predictions = np.repeat('00000000', len(out_labels))

df = {"ID" : list(par_ids),
      "Text" : list(text),
      "Predicted Labels" : list(out_predictions),
      "Gold Labels" : list(out_labels)}

df = pd.DataFrame(df)
df.to_csv("Inference_Results_all_zeros.csv", index=True)

metrics = compute_metrics(EvalPrediction(predictions, labels))

with open('metrics_all_zeros.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")
    
# all ones
#################
out_predictions = np.repeat('11111111', len(out_labels))

df = {"ID" : list(par_ids),
      "Text" : list(text),
      "Predicted Labels" : list(out_predictions),
      "Gold Labels" : list(out_labels)}

df = pd.DataFrame(df)
df.to_csv("Inference_Results_all_ones.csv", index=True)

metrics = compute_metrics(EvalPrediction(predictions, labels))

with open('metrics_all_ones.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")
    
    
# random
#################
out_predictions = [np.array2string(np.random.randint(0, 2, 8), separator='')[1:-1] for i in range(len(out_labels))]

df = {"ID" : list(par_ids),
      "Text" : list(text),
      "Predicted Labels" : list(out_predictions),
      "Gold Labels" : list(out_labels)}

df = pd.DataFrame(df)
df.to_csv("Inference_Results_random.csv", index=True)

metrics = compute_metrics(EvalPrediction(predictions, labels))

with open('metrics_random.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")