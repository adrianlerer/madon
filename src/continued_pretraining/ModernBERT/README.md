## Continued Pretraining ModernBERT
The trained model is currently located here: https://huggingface.co/TrustHLT/ModernBERT-large-czech-legal

Pretraining data was pulled from https://git-ce.rwth-aachen.de/trusthlt-public/czech-supreme-courts-data-and-crawler

## How to replicate the experiment
#### Create conda environment 
```bash
conda create --name modernbert python=3.12
conda activate modernbert
```
#### Install requirements
```bash
pip install -r requirements.txt
```
#### Install flash-attention 
```bash 
pip install packaging
pip install ninja
pip install flash-attn==2.7.4.post1 --no-build-isolation
```
#### (optional) Install comet for logging
```bash
pip install comet_ml
```
#### Preprocessing dataset
Make sure you have obtained and preprocessed the pretraining data following [these instructions](../README.md)
#### Train tokenizer
Set the pretraining data path and train the tokenizer
```bash
python continued_pretraining_modernbert_tokenizer.py
```
Warning: Overwriting the tokenizer will be similar to training the model from scratch
#### Run training
The training will run in 2 phases with different length of training data
```bash
python continued_pretraining_modernbert_model.py --model answerdotai/ModernBERT-large
```
Check the parameters for options

## Runtime stats
Phase 1: was trained on 2xA100 for ~36:00:00h

Phase 2: was trained on 2xA100 for ~01:00:00h