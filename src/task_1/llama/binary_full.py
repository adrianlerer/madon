from comet_ml import Experiment
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset, Dataset, DatasetDict
import os
import ast
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, hamming_loss
import pandas as pd
import random
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='llama_standard_binary_32')
    parser.add_argument('--training_type', type=str, default='standard')

    args = parser.parse_args()

    exp_name = args.output_dir
    model_path = args.model
    seed = args.seed
    output_dir = args.output_dir
    model_type = args.training_type

    experiment = Experiment(
        "Fill this with your Comet API key",
    )

    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic GPU operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)

    # === CONFIG ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4  # Reduced for long documents
    EPOCHS = 10 #5
    LEARNING_RATE = 1e-6

    # Binary classification labels
    
    label2id = {"No Argument": 0, "Argument": 1,}
    id2label = {0: "No Argument", 1: "Argument"}

    # === DATA PREP ===
    
    def process_data(split, quick=False):
        all_texts = []
        all_labels = []
        all_ids = []

        for example in split:
            doc_id = example['doc_id']
            paragraphs = ast.literal_eval(example['text'])
            label_list = example['labels']
            
            for i, para in enumerate(paragraphs):
                if quick and random.random() >= 0.03: # Quick mode
                    continue
                label = 1 if len(label_list[i]) > 0 else 0
                uid = f"{doc_id}_{i}"

                all_texts.append(para)
                all_labels.append(label)
                all_ids.append(uid)

        return {"text": all_texts, "labels": all_labels, "uid": all_ids}

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def tokenizer_function(data):
        return tokenizer(data["text"], truncation=True)

    # Load and process datasets
    dataset = load_dataset("TrustHLT/madon-init", name="default")
    
    train_data = Dataset.from_dict(process_data(dataset['train']))
    validation_data = Dataset.from_dict(process_data(dataset['validation']))
    test_data = Dataset.from_dict(process_data(dataset['test']))    
    
    
    train_data = train_data.map(tokenizer_function, batched=True)
    validation_data = validation_data.map(tokenizer_function, batched=True)
    test_data = test_data.map(tokenizer_function, batched=True)
    

    # set data format for PyTorch
    train_data.set_format("torch")
    validation_data.set_format("torch")
    test_data.set_format("torch")
    
    

    # === MODEL SETUP ===
    print("\nLoading model with classification head...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Data Collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    # === METRICS ===
    def compute_metrics(eval_pred):
        """Compute classification metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        rep = classification_report(
            labels, predictions, output_dict=True, zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'classification_report': rep
        }

    # === TRAINING SETUP ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        fp16=False,
        bf16=True,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=2,
        save_total_limit=2,
        report_to=["comet_ml"],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        remove_unused_columns=True,
        max_grad_norm=0.3,
        seed=seed,  # Using our seed here
        eval_accumulation_steps=None,
        torch_empty_cache_steps=3,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # === TRAINING ===
    print("\nStarting training...")
    trainer.train()


    # === PREDICTIONS ===
    final_model_path = os.path.join(output_dir, f"final_model-epoch-{EPOCHS}")
    trainer.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")

    print("\nEvaluating on test set...")
    logits, labels, metrics = trainer.predict(test_data)
    print(metrics)

    predictions = np.argmax(logits, axis=1)

    # Create results dataframe with UIDs
    results_df = pd.DataFrame({
        "ID": test_data["uid"],
        "Gold Labels": labels.tolist(),
        "Predicted Labels": predictions.tolist(),
    })

    results_df = results_df.sort_values("ID").reset_index(drop=True)

    # Add subset indicator to filename if using subset (For debugging)
    results_csv_path = os.path.join(output_dir, f"test_results_binary_llama_{model_type}_seed_{seed}.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    print(classification_report(labels, predictions))
    
    print(f"=== Completed experiment: {exp_name} ===\n")

