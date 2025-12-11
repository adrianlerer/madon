from datasets import load_dataset, Dataset
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import ast
from comet_ml import Experiment
import pandas as pd
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
import random

# === SEED CONFIGURATION FOR REPRODUCIBILITY ===
SEED = 2012

# Set all random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTHONHASHSEED"] = str(SEED)

# === CONFIG ===
MODEL_PATH = "meta-llama/Llama-3.1-8B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
EPOCHS = 5
MAX_LENGTH = 8000
CHUNK_OVERLAP = 256
LEARNING_RATE = 1e-5
OUTPUT_DIR = "./llama3_8B_standard_longdoc_classification" # This dir exists in SLURM
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Comet ML setup
experiment = Experiment(
"""Fill with your Comet API key"""
)
experiment.log_parameter("seed", SEED)

# Binary classification labels
BINARY_LABELS = {
    "O - OVERALL - NON FORMALISTIC": 0,
    "O - OVERALL - FORMALISTIC": 1,
}

# === DATA PROCESSING ===
print("Loading and chunking dataset...")

def chunk_document(text, label, tokenizer, doc_idx):
    """Chunk document into manageable pieces with overlap."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    for i in range(0, len(tokens), MAX_LENGTH - CHUNK_OVERLAP):
        chunk = tokens[i:i + MAX_LENGTH]
        chunks.append({
            "input_ids": chunk,
            "attention_mask": [1] * len(chunk),
            "labels": label,
            "chunk_id": f"{doc_idx}_{i}",
            "doc_idx": doc_idx
        })
    return chunks

def process_dataset(dataset, tokenizer):
    """Process raw dataset into chunked format."""
    processed = []
    for doc_idx, example in enumerate(dataset):
        text = " ".join(ast.literal_eval(example['text']))
        label = BINARY_LABELS[example['f_labels']]
        processed.extend(chunk_document(text, label, tokenizer, doc_idx))
    return processed

def create_dataset_dict(processed_data):
    """Convert processed data into dataset dictionary."""
    return {
        "input_ids": [x["input_ids"] for x in processed_data],
        "attention_mask": [x["attention_mask"] for x in processed_data],
        "labels": [x["labels"] for x in processed_data],
        "chunk_id": [x["chunk_id"] for x in processed_data],
        "doc_idx": [x["doc_idx"] for x in processed_data]
    }

def pad_sequences(examples):
    """Pad sequences to max length."""
    max_length = MAX_LENGTH
    padded_input_ids = []
    padded_attention_mask = []
    
    for input_ids, attention_mask in zip(examples["input_ids"], examples["attention_mask"]):
        if len(input_ids) < max_length:
            padding = [tokenizer.pad_token_id] * (max_length - len(input_ids))
            padded_input_ids.append(input_ids + padding)
            padded_attention_mask.append(attention_mask + [0] * (max_length - len(attention_mask)))
        else:
            padded_input_ids.append(input_ids[:max_length])
            padded_attention_mask.append(attention_mask[:max_length])
    
    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": examples["labels"]
    }

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load and process datasets
raw_dataset = load_dataset("TrustHLT/madon-init", name="experimental")
train_data = process_dataset(raw_dataset['train'], tokenizer)
test_data = process_dataset(raw_dataset['test'], tokenizer)

# Create datasets with all columns
full_train_dataset = Dataset.from_dict(create_dataset_dict(train_data))
full_test_dataset = Dataset.from_dict(create_dataset_dict(test_data))

# Create training datasets without string columns
train_dataset = full_train_dataset.map(
    pad_sequences,
    batched=True,
    batch_size=100,
    remove_columns=['chunk_id', 'doc_idx']  # Remove string columns
)

test_dataset = full_test_dataset.map(
    pad_sequences,
    batched=True,
    batch_size=100,
    remove_columns=['chunk_id', 'doc_idx']  # Remove string columns
)

# === MODEL SETUP ===
print("\nLoading model with classification head...")

# Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)

# The usual LoRA Configuration
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Data Collator
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=MAX_LENGTH,
    pad_to_multiple_of=8
)

# === THE USUAL METRICS ===
def compute_metrics(eval_pred):
    """Compute classification metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# === TRAINING SETUP ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    fp16=False,
    bf16=True,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    report_to=["comet_ml"],
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    remove_unused_columns=True,
    max_grad_norm=0.3,
    seed=SEED,  # Using our seed here
    dataloader_num_workers=4,
    eval_accumulation_steps=None,
    group_by_length=False
)

model.config.label2id = BINARY_LABELS
model.config.id2label = {v: k for k, v in BINARY_LABELS.items()}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# === TRAINING ===
print("\nStarting training...")
try:
    train_results = trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "best_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
    
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

except Exception as e:
    print(f"Training failed: {e}")
    trainer.save_model(os.path.join(OUTPUT_DIR, "interrupted_model"))
    raise

# === EVALUATION WITH REASSEMBLY ===
print("\nEvaluating with document-level reassembly...")

def reassemble_predictions(full_dataset, predictions):
    """Reassemble chunk predictions to document-level predictions using the full dataset with doc_idx."""
    doc_chunks = {}
    
    # Group chunks by document
    for i, chunk_data in enumerate(full_dataset):
        doc_idx = chunk_data['doc_idx']
        
        if doc_idx not in doc_chunks:
            doc_chunks[doc_idx] = {
                'chunks': [],
                'true_labels': None
            }
        
        doc_chunks[doc_idx]['chunks'].append({
            'prediction': predictions[i],
            'text': tokenizer.decode(chunk_data['input_ids'], skip_special_tokens=True)
        })
        
        # All chunks from same document have same label
        doc_chunks[doc_idx]['true_labels'] = chunk_data['labels']

    # Aggregate predictions
    doc_results = []
    for doc_idx, doc_data in doc_chunks.items():
        chunk_preds = [chunk['prediction'] for chunk in doc_data['chunks']]
        
        # Majority voting for document prediction
        doc_prediction = max(set(chunk_preds), key=chunk_preds.count)
        
        # Get original full text
        original_text = " ".join(ast.literal_eval(raw_dataset['test'][doc_idx]['text']))
        
        doc_results.append({
            'document_id': doc_idx,
            'text': original_text,
            'true_labels': doc_data['true_labels'],
            'predicted_labels': doc_prediction,
            'chunk_predictions': chunk_preds,
            'num_chunks': len(doc_data['chunks'])
        })
    
    return doc_results

# Get predictions
predictions = trainer.predict(test_dataset)
chunk_preds = np.argmax(predictions.predictions, axis=1)

# Reassemble to document level using the full dataset with doc_idx
document_results = reassemble_predictions(full_test_dataset, chunk_preds)


# Create a DataFrame with the required columns
results_df = pd.DataFrame({
    'Text': [doc['text'][:100] for doc in document_results],
    'Gold Labels': [doc['true_labels'] for doc in document_results],
    'Predicted Labels': [doc['predicted_labels'] for doc in document_results]
})

# Save to CSV
file_name = f"holistic_standard_{EPOCHS}epochs_{SEED}.csv"
doc_predictions_file = os.path.join(OUTPUT_DIR, file_name)
results_df.to_csv(doc_predictions_file, index=False)

# Calculate document-level metrics
true_labels = [doc['true_labels'] for doc in document_results]
pred_labels = [doc['predicted_labels'] for doc in document_results]

accuracy = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average='binary', zero_division=0
)

document_metrics = {
    'document_accuracy': accuracy,
    'document_precision': precision,
    'document_recall': recall,
    'document_f1': f1
}

print("\nDocument-Level Metrics:")
print(document_metrics)

# Save combined metrics
trainer.save_metrics("eval", {
    **predictions.metrics,  # chunk-level metrics
    **document_metrics      # document-level metrics
})

print("\nEvaluation complete!")
print(f"Document predictions saved to {doc_predictions_file}")