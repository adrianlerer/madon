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

# FIXED: Enable full determinism with the seed
enable_full_determinism(SEED)

from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
import evaluate
import numpy as np
import ipdb
import pandas as pd
from accelerate import Accelerator
import accelerate 

# FIXED: Set seeds for all libraries
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

#########
# Logging
#########

push_to_hub = True

experiment = start(
  api_key="your-api-key-in-here",
  project_name="your-project-name-in-here",
  workspace="your-workspace-in-here"
)

#########

#############
# Hyperparams
#############
num_epochs      = 60
batch_size      = 4
accum_steps     = 6         # if GPU memory is tight, you might set this to 2, 3, 4..
learning_rate   = 1e-6
weight_decay    = 0.001
warmup_ratio    = 0.0
dropout         = 0.1
checkpoint      = "answerdotai/ModernBERT-large"
checkpoint_dest = "where-to-save-the-model-on-huggingface"
dataset_id      = "your-dataset-path-on-huggingface"
num_labels      = 2
resume_from_checkpoint     = False
max_input_length= 30000
#############


############
# Load model
############
accelerator = Accelerator(mixed_precision="bf16")
device = accelerator.device

config = AutoConfig.from_pretrained(
    checkpoint,
    classifier_dropout=dropout
    )

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config).to(device)
##########################


######################
# Dataset + Tokenizer
######################
dataset = load_dataset(dataset_id)

tokenizer = AutoTokenizer.from_pretrained(checkpoint, max_length=max_input_length) # , reference_compile=False)
def tokenize_function(data):
    return tokenizer(data["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text']).with_format("torch")

data_collator = DataCollatorWithPadding(tokenizer)
######################


########################
# Some Training helpers
########################

def compute_metrics(eval_pred):
    #ipdb.set_trace()
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    
    # precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    # accuracy = accuracy_score(labels, predictions)
    
    # return {
    #     'accuracy': accuracy,
    #     'precision': precision,
    #     'recall': recall,
    #     'f1': f1
    # }

    return { "classification_report": classification_report(labels, predictions, output_dict=True) } # , target_names=label2id.keys()) 

training_args = TrainingArguments(
    output_dir="overall_binary",
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
    bf16=True,
    
    # logging and eval
    save_strategy="epoch", 
    eval_strategy="epoch"
    # save_total_limit=3,
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
    # load_best_model_at_end=True
  )

trainer = accelerator.prepare(Trainer(
    model=model, 
    args=training_args, 
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator, 
    compute_metrics=compute_metrics
    ))

########################

##########
# Launch Training
##########
trainer.train()
trainer.push_to_hub()

##########