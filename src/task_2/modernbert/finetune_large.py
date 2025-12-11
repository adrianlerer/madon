# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:39:06 2025

@author: harun

To run this:
    accelerate config
    accelerate launch train.py
"""

from comet_ml import start
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoConfig, enable_full_determinism
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

###########################
# Setting the Random Seed
###########################


SEED = 42

enable_full_determinism(SEED)


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
import accelerate

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

###########################

#########
# Logging
#########

experiment = start(
  api_key="your-api-key-in-here",
  project_name="your-project-name-in-here",
  workspace="your-workspace-in-here"
)

push_to_hub = True

#########

#############
# Hyperparams
#############
num_epochs      = 50
batch_size      = 16
accum_steps     = 1         # if GPU memory is tight, you might set this to 2, 3, 4..
learning_rate   = 10e-5
weight_decay    = 0.00
warmup_ratio    = 0.00
dropout         = 0.0
checkpoint_src  = "answerdotai/ModernBERT-large" # there is also a large version with 2 times parameters
revision_id     = "main"
checkpoint_dest = "where-to-save-the-model-on-huggingface"
dataset_id      = "your-dataset-path-on-huggingface"
num_labels      = 8
max_input_length= 3000
#############

dataset = load_dataset(dataset_id)
label2id = {'LIN': 0, 'SI': 1, 'CL': 2, 'D': 3, 'HI': 4, 'PL': 5, 'TI': 6, 'PC': 7}
id2label = {0: 'LIN', 1: 'SI', 2: 'CL', 3: 'D', 4: 'HI', 5: 'PL', 6: 'TI', 7: 'PC'}

tokenizer = AutoTokenizer.from_pretrained(checkpoint_src, max_length=max_input_length)

def tokenize_function(data):
    return tokenizer(data["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer)
        

##########################

############
# Load model
############

accelerator = Accelerator(mixed_precision="bf16")
accelerate.utils.set_seed(SEED, device_specific=True, deterministic=True)
device = accelerator.device

config = AutoConfig.from_pretrained(
    checkpoint_src,
    classifier_dropout=dropout,
    num_labels=num_labels,
    label2id = label2id,
    id2label=id2label,
    problem_type="multi_label_classification"
    )

def model_init():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    return AutoModelForSequenceClassification.from_pretrained(checkpoint_src, config=config, revision=revision_id)

########################
# Some Training helpers
########################

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # first calculate the predictions given a threshold
    threshold = 0.5
    predictions = (expit(logits) >= threshold).astype(int)
    
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


training_args = TrainingArguments(
    output_dir="multilabel_paragraph",
    hub_model_id=checkpoint_dest,
    push_to_hub=push_to_hub,
    report_to="comet_ml",

    # data and batching
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=accum_steps,
    
    # optimization
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    warmup_ratio=warmup_ratio,
    num_train_epochs=num_epochs,
    
    # Random seed
    seed=SEED,
    data_seed=SEED,
    
    # logging and eval
    save_strategy="epoch", 
    eval_strategy="epoch"
  )


trainer = accelerator.prepare(Trainer(
    model_init=model_init,
    args=training_args, 
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator, 
    compute_metrics=compute_metrics
    ))

########################

##########
# Training
##########
trainer.train()

if push_to_hub:
    trainer.push_to_hub()

##########