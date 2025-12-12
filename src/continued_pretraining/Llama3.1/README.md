## Continued Pretraining Llama 3.1 8B
This code was used to continue pretraining the Llama 3.1 8B base model with Czech legal data.

The trained model is currently located here: https://huggingface.co/TrustHLT/Llama-3.1-8B-czech-legal

Pretraining data was pulled from https://git-ce.rwth-aachen.de/trusthlt-public/czech-supreme-courts-data-and-crawler

## How to replicate the experiment

#### Create conda environment 
```bash
conda create --name llama python=3.12
conda activate llama
```
#### Install requirements
```bash
pip install -r requirements.txt
```
#### (optional) Install comet for logging
```bash
pip install comet_ml
```
#### Preprocessing dataset
Make sure you have obtained and preprocessed the pretraining data following [these instructions](../README.md)

#### Run training
```bash
python continued_pretraining_llama31_model.py --model unsloth/Meta-Llama-3.1-8B
```
Check the parameters for options

## Runtime stats

The model was trained on 1xA100 for ~32:00:00h