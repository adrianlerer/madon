# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:59:32 2025

@author: harun
"""

from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset_src = "HUGGING_FACE_KEY_FOR_MADON_DATASET"
config = "HUGGING_FACE_CONFIG_FOR_MADON_DATASET"
dataset_dest = "HUGGING_FACE_DESTINATION_FOR_PREPROCESSED_DATASET"


checkpoint = "answerdotai/ModernBERT-large"
num_labels = 2
label2id = {'O - OVERALL - NON FORMALISTIC': 0, 'O - OVERALL - FORMALISTIC': 1}

test = False

##########################
# Load and preprocess data
##########################
raw_datasets = load_dataset(dataset_src, config)


# max_input_length = 60000
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(data):
    return tokenizer(data["text"]) # max input length is 2799

# encode the string label to an integer
def encode_labels(example):
    example["f_labels"] = label2id[example["f_labels"]]
    return example

# create a single string of the whole document
def encode_text(example):
    text = ast.literal_eval(example["text"])
    example["text"] = [''.join(text) for i in range(len(text))][0]
    return example

encoded_labels_dataset = raw_datasets.map(encode_labels)
encoded_text_dataset = encoded_labels_dataset.map(encode_text)
encoded_text_dataset = encoded_text_dataset.remove_columns(['labels', 'result_labels'])
encoded_text_dataset = encoded_text_dataset.rename_column("f_labels", "labels")
encoded_text_dataset = encoded_text_dataset.cast_column("labels", ClassLabel(num_classes=1))
encoded_text_dataset.set_format("torch")

encoded_text_dataset["dev"] = encoded_text_dataset["validation"]
del encoded_text_dataset["validation"]

# Test the dataloaders working
if test:
    # set_lens = [189, 29, 54]
    # names = ["train", "test", "dev"]
    # for i in range(3):
    #     for k in range(set_lens[i]):
    #         assert dataset[names[i]][k]["text"] == encoded_text_dataset[names[i]][k]["text"]
    #         assert dataset[names[i]][k]["labels"] == encoded_text_dataset[names[i]][k]["labels"]
    #         print(k)
    #         # print(encoded_text_dataset[names[i]][k]["doc_id"])
    
    tokenized_datasets = encoded_text_dataset.map(tokenize_function, batched=True)

    train_lengths = []
    for i, batch in enumerate(tokenized_datasets["train"]):
        #print({k: v.shape for k, v in batch.items()})
        train_lengths.append(batch["input_ids"].shape[0])
    
    plt.figure()
    plt.hist(train_lengths, bins=30, color="#69b3a2", edgecolor="black")
    plt.title("Train Sequence Lengths", fontsize=20, fontweight="bold")
    plt.xlabel("Input Sequence Length", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    val_lengths = []
    for i, batch in enumerate(tokenized_datasets["dev"]):
        #print({k: v.shape for k, v in batch.items()})
        val_lengths.append(batch["input_ids"].shape[0])
    
    plt.figure()
    plt.hist(val_lengths, bins=30, color="#69b3a2", edgecolor="black")
    plt.title("Validation Sequence Lengths", fontsize=20, fontweight="bold")
    plt.xlabel("Input Sequence Length", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    test_lengths = []
    for i, batch in enumerate(tokenized_datasets["test"]):
        #print({k: v.shape for k, v in batch.items()})
        test_lengths.append(batch["input_ids"].shape[0])
    
    plt.figure()
    plt.hist(test_lengths, bins=30, color="#69b3a2", edgecolor="black")
    plt.title("Test Sequence Lengths", fontsize=20, fontweight="bold")
    plt.xlabel("Input Sequence Length", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    all_lengths =  train_lengths + val_lengths + test_lengths
    plt.figure()
    plt.hist(all_lengths, bins=30, color="#69b3a2", edgecolor="black")
    plt.title("All Sequence Lengths", fontsize=20, fontweight="bold")
    plt.xlabel("Input Sequence Length", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    
else:    
    encoded_text_dataset.push_to_hub(dataset_dest)
