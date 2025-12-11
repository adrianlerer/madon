# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:59:32 2025

@author: harun
"""

from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
import ast
import numpy as np
import pandas as pd

dataset_src = "HUGGING_FACE_KEY_FOR_MADON_DATASET"
config = "HUGGING_FACE_CONFIG_FOR_MADON_DATASET"
dataset_dest = "HUGGING_FACE_DESTINATION_FOR_PREPROCESSED_DATASET"


checkpoint = "answerdotai/ModernBERT-large"
##########################
# Load and preprocess data
##########################
raw_datasets = load_dataset(dataset_src, config)


print("Rearranging train dataset to train paragraph-wise")
data_type = "train"
labels = np.array([])
par_ids = np.array([])
paragraphs = []
num_documents = len(raw_datasets[data_type])
for i in range(num_documents):
    num_paragraphs = len(raw_datasets[data_type][i]["result_labels"])
    par_labels = np.array([len(raw_datasets[data_type][i]["labels"][k]) > 0 for k in range(num_paragraphs)], dtype=np.int8)
    labels = np.concatenate((labels, par_labels), axis=0)
    paragraphs += ast.literal_eval(raw_datasets[data_type][i]["text"])
    par_id = np.array([raw_datasets[data_type][i]["doc_id"] + "_" + str(k) for k in range(num_paragraphs)])
    par_ids = np.concatenate((par_ids, par_id), axis=0)
    
train_pd = pd.DataFrame({"par_ids": par_ids, "text": paragraphs, "labels": labels})

print("Rearranging validation dataset to train paragraph-wise")
data_type = "validation"
labels = np.array([])
par_ids = np.array([])
paragraphs = []
num_documents = len(raw_datasets[data_type])
for i in range(num_documents):
    num_paragraphs = len(raw_datasets[data_type][i]["result_labels"])
    par_labels = np.array([len(raw_datasets[data_type][i]["labels"][k]) > 0 for k in range(num_paragraphs)], dtype=np.int8)
    labels = np.concatenate((labels, par_labels), axis=0)
    paragraphs += ast.literal_eval(raw_datasets[data_type][i]["text"])
    par_id = np.array([raw_datasets[data_type][i]["doc_id"] + "_" + str(k) for k in range(num_paragraphs)])
    par_ids = np.concatenate((par_ids, par_id), axis=0)  

val_pd = pd.DataFrame({"par_ids": par_ids, "text": paragraphs, "labels": labels})

print("Rearranging test dataset to test paragraph-wise")
data_type = "test"
labels = np.array([])
par_ids = np.array([])
paragraphs = []
num_documents = len(raw_datasets[data_type])
for i in range(num_documents):
    num_paragraphs = len(raw_datasets[data_type][i]["result_labels"])
    par_labels = np.array([len(raw_datasets[data_type][i]["labels"][k]) > 0 for k in range(num_paragraphs)], dtype=np.int8)
    labels = np.concatenate((labels, par_labels), axis=0)
    paragraphs += ast.literal_eval(raw_datasets[data_type][i]["text"])
    par_id = np.array([raw_datasets[data_type][i]["doc_id"] + "_" + str(k) for k in range(num_paragraphs)])
    par_ids = np.concatenate((par_ids, par_id), axis=0)  

test_pd = pd.DataFrame({"par_ids": par_ids, "text": paragraphs, "labels": labels})

train_dataset = Dataset.from_pandas(train_pd)
test_dataset = Dataset.from_pandas(test_pd)
valid_dataset = Dataset.from_pandas(val_pd)

# Create a DatasetDict
preprocessed_dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "dev": valid_dataset
})

preprocessed_dataset = preprocessed_dataset.cast_column("labels", ClassLabel(num_classes=2))

# set_lens = [6246, 1085, 1852]
# names = ["train", "test", "dev"]
# for i in range(3):
#     for k in range(set_lens[i]):
#         assert dataset[names[i]][k]["text"] == preprocessed_dataset[names[i]][k]["text"]
#         assert dataset[names[i]][k]["labels"] == preprocessed_dataset[names[i]][k]["labels"]
#         print(k)

preprocessed_dataset.push_to_hub(dataset_dest)


















