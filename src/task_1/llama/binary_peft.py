from comet_ml import Experiment
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, PeftModelForSequenceClassification
from datasets import load_dataset, Dataset
import os
import ast
import numpy as np
from sklearn.metrics import classification_report
import glob
import pandas as pd
import random

# === SEED FOR REPRODUCIBILITY ===
SEED = "choose your seed"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# === CONFIG ===
MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 5
MAX_LENGTH = 4000
LEARNING_RATE = 2e-5
OUTPUT_DIR = f"./llama3_bce_standard_{SEED}"
PENALTY_WEIGHT = 2.0  # Penalty multiplier for false positives/negatives

# === DATA PREP ===
print("Loading dataset...")
dataset = load_dataset("TrustHLT/madon-init", name="default")

def process_data(split, balance=False):
    texts_0 = []  
    texts_1 = [] 
    labels_0 = []
    labels_1 = []
    
    for example in split:
        paragraphs = ast.literal_eval(example['text'])
        label_list = example['labels']
        for i, para in enumerate(paragraphs):
            if len(label_list[i]) > 0:  # Class 1
                texts_1.append(para)
                labels_1.append(1)
            else:  # Class 0
                texts_0.append(para)
                labels_0.append(0)
    
    if balance:
        min_samples = int(min(len(texts_1), len(texts_0))/5)
        balanced_texts = texts_1[:min_samples] + texts_0[:min_samples]
        balanced_labels = labels_1[:min_samples] + labels_0[:min_samples]
        return {"text": balanced_texts, "labels": balanced_labels}
    else:
        return {"text": texts_0 + texts_1, "labels": labels_0 + labels_1}

train_data = Dataset.from_dict(process_data(dataset['train'], balance=False))
test_data = Dataset.from_dict(process_data(dataset['test'], balance=False))

# Calculate class weights for BCE loss (important for imbalanced datasets)

pos_weight = torch.tensor([len(train_data['labels']) / sum(train_data['labels']) - 1], 
                         dtype=torch.float32).to(DEVICE)


# === MODEL SETUP ===
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "left"

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    max_length=MAX_LENGTH,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    ignore_mismatched_sizes=True
)

with torch.no_grad():
    torch.nn.init.normal_(model.score.weight, mean=0.0, std=0.02)
    if hasattr(model.score, 'bias') and model.score.bias is not None:
        model.score.bias.zero_()

model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

model = PeftModelForSequenceClassification(model, peft_config)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

train_dataset = train_data.map(tokenize_function, batched=True)
test_dataset = test_data.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# === Trainer using BCE and penalties for false positives/negatives === # 
class PenalizedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        
        logits = logits.view(-1)  
        labels = labels.view(-1)       
        
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        losses = loss_fct(logits, labels)
        
        # Apply penalties
        preds = (torch.sigmoid(logits) > 0.5).float()
        preds = preds.to(labels.device)  
        
        false_positives = (labels == 0) & (preds == 1)
        false_negatives = (labels == 1) & (preds == 0)
        
        losses[false_positives] *= PENALTY_WEIGHT
        losses[false_negatives] *= PENALTY_WEIGHT
        
        return (losses.mean(), outputs) if return_outputs else losses.mean()

training_args = TrainingArguments(
    max_grad_norm=1.0,
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    bf16=True,
    save_total_limit=2,
    report_to=["comet_ml"],
    gradient_checkpointing=True,
    optim="adamw_torch",
    gradient_accumulation_steps=2,
    remove_unused_columns=False,
    label_names=["labels"],
    warmup_ratio=0.1,
)

def compute_metrics(p):
    predictions, labels = p
    predictions = torch.sigmoid(torch.tensor(predictions)).squeeze() 
    pred_labels = (predictions > 0.5).int().numpy()
    labels = labels.astype(int)
    return {
        "accuracy": (pred_labels == labels).mean(),
        "classification_report": classification_report(labels, pred_labels, 
                                                     target_names=["class_0", "class_1"], 
                                                     output_dict=True)
    }

trainer = PenalizedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

final_model_path = os.path.join(OUTPUT_DIR, f"final_model-epoch-{EPOCHS}")
trainer.save_model(final_model_path)
print(f"Final model saved to {final_model_path}")

print("\nEvaluating on test set...")
predictions = trainer.predict(test_dataset)
pred_probs = torch.sigmoid(torch.tensor(predictions.predictions)).squeeze()
pred_labels = (pred_probs > 0.5).int().numpy()
true_labels = predictions.label_ids.astype(int)

# df with results
results_df = pd.DataFrame({
    "Text": test_data["text"],
    "Gold_Labels": true_labels,
    "Predicted_Labels": pred_labels,
    "Prediction_Probabilities": pred_probs.numpy()
})

# save the df as a csv file
results_csv_path = os.path.join(OUTPUT_DIR, f"test_results-{EPOCHS}.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")

print("\nClassification Report:")

print(classification_report(true_labels, pred_labels, target_names=["class_0", "class_1"]))