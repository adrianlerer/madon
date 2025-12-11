# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:52:13 2025

@author: harun
"""

from comet_ml import start
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
import evaluate
import numpy as np
import ipdb
import pandas as pd
from accelerate import Accelerator

model_id = "USERNAME/MODEL_LINK_ON_HUGGINGFACE" 
revision_id = "THE_SPECIFIC_REVISION_ID_TO_USE" # "main"
dataset_id = "USERNAME/PREPROCESSED_DATASET_ON_HUGGINGFACE"

accelerator = Accelerator(mixed_precision="bf16")

model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision_id, reference_compile=False)

dataset = load_dataset(dataset_id)

max_input_length = 30000
tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=max_input_length)

def tokenize_function(data):
    return tokenizer(data["text"], truncation=True) 

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer)

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

batch_size = 1

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
doc_ids = tokenized_datasets["test"]["doc_id"]
# text = dataset["test"]["text"]
# if i include the text, excel has a problem with displaying it correctly (probably something with delimiter or so. Yassines evaluation script is also not working correctly in that case)
# ipdb.set_trace()


predictions = np.argmax(logits, axis=-1)

df = {"ID" : list(doc_ids),
      # "Text" : list(text),
      "Logits" : list(logits),
      "Predicted Labels" : predictions,
      "Gold Labels" : labels}


df = pd.DataFrame(df)
df.to_csv("Inference_Results.csv", index=True)

conf_mat = confusion_matrix(labels, predictions, labels=[0, 1])

with open('metrics.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")
    f.write("\nConfusion Matrix: \n" + str(conf_mat))
    

# all zeros
####################
predictions = np.zeros(labels.shape)


df = {"ID" : list(doc_ids),
      # "Text" : list(text),
      "Predicted Labels" : predictions,
      "Gold Labels" : labels}

df = pd.DataFrame(df)
df.to_csv("Inference_Results_all_zeros.csv", index=True)

metrics = {"classification_report": classification_report(labels, predictions, output_dict=True)}

conf_mat = confusion_matrix(labels, predictions, labels=[0, 1])

with open('metrics_all_zeros.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")
    f.write("\nConfusion Matrix: \n" + str(conf_mat))
    
# all ones
#################
predictions = np.ones(labels.shape)

df = {"ID" : list(doc_ids),
      # "Text" : list(text),
      "Predicted Labels" : predictions,
      "Gold Labels" : labels}

df = pd.DataFrame(df)
df.to_csv("Inference_Results_all_ones.csv", index=True)

metrics = {"classification_report": classification_report(labels, predictions, output_dict=True)}

conf_mat = confusion_matrix(labels, predictions, labels=[0, 1])

with open('metrics_all_ones.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")
    f.write("\nConfusion Matrix: \n" + str(conf_mat))
    
    
# random
#################
predictions = np.random.randint(0, 2, labels.shape)

df = {"ID" : list(doc_ids),
      # "Text" : list(text),
      "Predicted Labels" : predictions,
      "Gold Labels" : labels}

df = pd.DataFrame(df)
df.to_csv("Inference_Results_random.csv", index=True)

metrics = {"classification_report": classification_report(labels, predictions, output_dict=True)}

conf_mat = confusion_matrix(labels, predictions, labels=[0, 1])

with open('metrics_random.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")
    f.write("\nConfusion Matrix: \n" + str(conf_mat))