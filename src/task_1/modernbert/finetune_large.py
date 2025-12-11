# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:39:06 2025

@author: harun

To run this:
    accelerate config
    accelerate launch large.py
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

###########################

import numpy as np
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import classification_report
from accelerate import Accelerator
import accelerate

# FIXED: Set seeds for all libraries
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


#########
# Logging
#########

test = False
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
num_epochs      = 50 # 50
batch_size      = 16 # 256
accum_steps     = 1         # if GPU memory is tight, you might set this to 2, 3, 4..
learning_rate   = 5e-6
weight_decay    = 0.01
warmup_ratio    = 0.00
dropout         = 0.0
checkpoint_src  = "answerdotai/ModernBERT-large"
revision_id     = "main"
checkpoint_dest = "where-to-save-the-model-on-huggingface"
dataset_id      = "your-dataset-path-on-huggingface"
num_labels      = 2
max_input_length= 3000
#############

dataset = load_dataset(dataset_id)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_src, max_length=max_input_length)
def tokenize_function(data):
    return tokenizer(data["text"], truncation=True) # max input length is 2799

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
    )

def model_init():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    return AutoModelForSequenceClassification.from_pretrained(checkpoint_src, config=config, revision=revision_id)

############

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

    return {
        "classification_report": classification_report(labels, predictions, output_dict=True) # , target_names=label2id.keys())
    }

training_args = TrainingArguments(
    output_dir="binary_paragraph",
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
    
    seed=SEED,
    data_seed=SEED,
        
    # logging and eval
    save_strategy="epoch",
    eval_strategy="epoch"
    )

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # build your weight tensor one time
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 6.42], device=device))

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

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
trainer.push_to_hub()

##########