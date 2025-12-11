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
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pandas as pd
import random

# === EXPERIMENTS CONFIG ===
experiments = {
    "standard_32": ["meta-llama/Meta-Llama-3-8B", 32, "llama_standard_multi_label_32_asy", "standard"],
    "standard_42": ["meta-llama/Meta-Llama-3-8B", 42, "llama_standard_multi_label_42_asy", "standard"],
    "standard_52": ["meta-llama/Meta-Llama-3-8B", 52, "llama_standard_multi_label_52_asy", "standard"],
    "cpt_32": ["TrustHLT-ECALP/llama-3.1-8b-cpt-full", 32, "llama_cpt_multi_label_32_asy", "cpt"],
    "cpt_42": ["TrustHLT-ECALP/llama-3.1-8b-cpt-full", 42, "llama_cpt_multi_label_42_asy", "cpt"],
    "cpt_52": ["TrustHLT-ECALP/llama-3.1-8b-cpt-full", 52, "llama_cpt_multi_label_52_asy", "cpt"],
}

# === GLOBAL CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 5
MAX_LENGTH = 4000
LEARNING_RATE = 2e-5
PENALTY_WEIGHT = 2.0
ARGUMENT_LABELS = ["LIN", "SI", "CL", "D", "HI", "PL", "TI", "PC"]
USE_SUBSET = False  # Set to True to use only 1/20th of the data

# === DATA PREPARATION (done once) ===
print("Loading dataset...")
dataset = load_dataset("TrustHLT/madon-init", name="default")

def process_data(split, subset=False):
    all_texts = []
    all_labels = []
    all_ids = []

    for example in split:
        doc_id = example['doc_id']
        paragraphs = ast.literal_eval(example['text'])
        label_list = example['labels']
        
        for i, para in enumerate(paragraphs):
            # For Debugging: Skip 19 out of 20 examples if subset is True (keeping only 5% of data)
            if subset and random.random() >= 0.01:
                continue
                
            label_str = mapping_labels(label_list[i])
            uid = f"{doc_id}_{i}"

            all_texts.append(para)
            all_labels.append(label_str)
            all_ids.append(uid)

    return {"text": all_texts, "labels": all_labels, "uid": all_ids}

def mapping_labels(labels):
    """Convert list of labels to binary string"""
    label_str = ""
    for label in ARGUMENT_LABELS:
        label_str += "1" if label in labels else "0"
    return label_str

# Process data with UIDs and optional subset
train_data = Dataset.from_dict(process_data(dataset['train'], subset=USE_SUBSET))
test_data = Dataset.from_dict(process_data(dataset['test'], subset=USE_SUBSET))

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# Convert to binary vectors
def labels_to_binary(label_strings):
    return np.array([[int(c) for c in s] for s in label_strings])

train_labels = labels_to_binary(train_data['labels'])
test_labels = labels_to_binary(test_data['labels'])

#### Not Used in this script, but kept for reference ####

# # Calculate class weights for BCE loss (using training data)
# class_weights = []
# for i in range(len(ARGUMENT_LABELS)):
#     pos_count = train_labels[:, i].sum()
#     neg_count = len(train_labels) - pos_count
#     weight = neg_count / (pos_count + 1e-5)
#     class_weights.append(weight)

# class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# === EXPERIMENT LOOP ===
for exp_name, (model_path, seed, output_dir, model_type) in experiments.items():
    print(f"\n=== Starting experiment: {exp_name} (Seed: {seed}) ===")
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # === MODEL SETUP ===
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
        model_path,
        num_labels=len(ARGUMENT_LABELS),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        ignore_mismatched_sizes=True
    )

    # Initialize classification head
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
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        # Keep uid and text as non-tensor metadata
        tokenized["uid"] = examples["uid"]
        tokenized["text"] = examples["text"]
        return tokenized

    # Tokenize datasets
    train_dataset = train_data.map(tokenize_function, batched=True)
    test_dataset = test_data.map(tokenize_function, batched=True)

    # Add binary labels
    train_dataset = train_dataset.add_column("binary_labels", list(train_labels))
    test_dataset = test_dataset.add_column("binary_labels", list(test_labels))

    # Format datasets for Trainer, exclude uid/text from tensor formatting
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "binary_labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "binary_labels"])

    class AsymmetricLoss(torch.nn.Module):
        """
        Asymmetric Loss for multi-label classification.
        """
        def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
            super(AsymmetricLoss, self).__init__()
            self.gamma_neg = gamma_neg
            self.gamma_pos = gamma_pos
            self.clip = clip
            self.eps = eps

        def forward(self, logits, targets):
            """Logits: raw model outputs. Targets: binary labels."""
            logits_sigmoid = torch.sigmoid(logits)
            targets = targets.float()

            if self.clip is not None and self.clip > 0:
                logits_sigmoid = torch.clamp(logits_sigmoid, min=self.clip, max=1 - self.clip)

            pos_loss = targets * torch.log(logits_sigmoid + self.eps)
            neg_loss = (1 - targets) * torch.log(1 - logits_sigmoid + self.eps)

            pos_weight = (1 - logits_sigmoid) ** self.gamma_pos
            neg_weight = logits_sigmoid ** self.gamma_neg

            loss = - (pos_weight * pos_loss + neg_weight * neg_loss)
            return loss.mean()


    # === Trainer using BCE with class weights ===
    class MultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("binary_labels").float()
            outputs = model(**inputs)
            logits = outputs.logits

            loss_fct = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
            loss = loss_fct(logits, labels)

            return (loss, outputs) if return_outputs else loss


    training_args = TrainingArguments(
        max_grad_norm=1.0,
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        bf16=True,
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="adamw_torch",
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
        label_names=["binary_labels"],
        warmup_ratio=0.1,
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        pred_labels = (predictions > 0.5).astype(int)
        
        results = {}
        for i, label in enumerate(ARGUMENT_LABELS):
            results[f"f1_{label}"] = f1_score(labels[:, i], pred_labels[:, i], zero_division=0)
        
        results["f1_micro"] = f1_score(labels, pred_labels, average='micro')
        results["f1_macro"] = f1_score(labels, pred_labels, average='macro')
        results["accuracy"] = accuracy_score(labels, pred_labels)
        return results

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    final_model_path = os.path.join(output_dir, f"final_model-epoch-{EPOCHS}")
    trainer.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")

    print("\nEvaluating on test set...")
    predictions = trainer.predict(test_dataset)
    pred_probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
    pred_labels = (pred_probs > 0.5).astype(int)
    true_labels = predictions.label_ids

    # Create results dataframe with UIDs
    results = []
    for i in range(len(test_dataset)):
        uid = test_data[i]["uid"]
        text = test_data[i]["text"]
        true_str = test_data[i]["labels"]
        pred_str = "".join(str(b) for b in pred_labels[i])
        
        label_results = {}
        for j, label in enumerate(ARGUMENT_LABELS):
            label_results[f"{label}_prob"] = pred_probs[i][j]
        
        results.append({
            "UID": uid,
            "Text": text,
            "Gold Labels": true_str,
            "Predicted Labels": pred_str,
            **label_results
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("UID").reset_index(drop=True)

    # Add subset indicator to filename if using subset (For debugging)
    subset_suffix = "_subset" if USE_SUBSET else ""
    results_csv_path = os.path.join(output_dir, f"test_results-{EPOCHS}_seed_{seed}_{model_type}{subset_suffix}.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    # Log classification reports (for quick reference)
    print("\nClassification Reports:")
    for label in ARGUMENT_LABELS:
        idx = ARGUMENT_LABELS.index(label)
        report = classification_report(
            true_labels[:, idx],
            pred_labels[:, idx],
            target_names=["absent", "present"],
            output_dict=True
        )
        print(f"\n{label}:")
        print(classification_report(
            true_labels[:, idx],
            pred_labels[:, idx],
            target_names=["absent", "present"]
        ))
        
    # Log overall metrics (For quick reference)
    micro_f1 = f1_score(true_labels, pred_labels, average='micro')
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    accuracy = accuracy_score(true_labels, pred_labels)
    
    print("\nOverall Metrics:")
    print(f"Micro F1: {micro_f1}")
    print(f"Macro F1: {macro_f1}")
    print(f"Accuracy: {accuracy}")
    
    print(f"=== Completed experiment: {exp_name} ===\n")