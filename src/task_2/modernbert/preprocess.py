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
num_labels = 8
label2id = {'LIN': 0, 'SI': 1, 'CL': 2, 'D': 3, 'HI': 4, 'PL': 5, 'TI': 6, 'PC': 7}

##########################
# Load and preprocess data
##########################
raw_datasets = load_dataset(dataset_src, config)


###############
print("Rearranging train dataset to train paragraph-wise")
data_type = "train"
labels = []
par_ids = np.array([])
paragraphs = []
num_documents = len(raw_datasets[data_type])
for i in range(num_documents):
    num_paragraphs = len(raw_datasets[data_type][i]["result_labels"])
    labels += raw_datasets[data_type][i]["labels"]
    paragraphs += ast.literal_eval(raw_datasets[data_type][i]["text"])
    par_id = np.array([raw_datasets[data_type][i]["doc_id"] + "_" + str(k) for k in range(num_paragraphs)])
    par_ids = np.concatenate((par_ids, par_id), axis=0)
    
train_pd = pd.DataFrame({"par_ids": par_ids, "text": paragraphs, "labels": labels})

print("Rearranging validation dataset to train paragraph-wise")
data_type = "validation"
labels = []
par_ids = np.array([])
paragraphs = []
num_documents = len(raw_datasets[data_type])
for i in range(num_documents):
    num_paragraphs = len(raw_datasets[data_type][i]["result_labels"])
    labels += raw_datasets[data_type][i]["labels"]
    paragraphs += ast.literal_eval(raw_datasets[data_type][i]["text"])
    par_id = np.array([raw_datasets[data_type][i]["doc_id"] + "_" + str(k) for k in range(num_paragraphs)])
    par_ids = np.concatenate((par_ids, par_id), axis=0)

val_pd = pd.DataFrame({"par_ids": par_ids, "text": paragraphs, "labels": labels})

print("Rearranging test dataset to test paragraph-wise")
data_type = "test"
labels = []
par_ids = np.array([])
paragraphs = []
num_documents = len(raw_datasets[data_type])
for i in range(num_documents):
    num_paragraphs = len(raw_datasets[data_type][i]["result_labels"])
    labels += raw_datasets[data_type][i]["labels"]
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


def encode_labels(example):
    raw = example['labels']
    example['labels'] = np.zeros((num_labels))
    for i in range(len(raw)):
        example['labels'][label2id[raw[i]]] = 1
    return example

preprocessed_dataset = preprocessed_dataset.map(encode_labels)

# set_lens = [6246, 1085, 1852]
# names = ["train", "test", "dev"]
# for i in range(3):
#     for k in range(set_lens[i]):
#         assert dataset[names[i]][k]["text"] == preprocessed_dataset[names[i]][k]["text"]
#         assert dataset[names[i]][k]["labels"] == preprocessed_dataset[names[i]][k]["labels"]
#         print(k)

preprocessed_dataset.push_to_hub(dataset_dest)
