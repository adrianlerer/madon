import comet_ml
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='llama_standard_multi_label_42')
    parser.add_argument('--training_type', type=str, default='standard')

    args = parser.parse_args()

    exp_name = args.output_dir
    model_path = args.model
    seed = args.seed
    output_dir = args.output_dir
    model_type = args.training_type

    # experiment = comet_ml.start(
    #         workspace="madon",
    #         project_name="llama",
    #         experiment_config=comet_ml.ExperimentConfig(
    #             name=exp_name,
    #             parse_args=True
    #         ),
    #     )

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
    BATCH_SIZE = 1  # Reduced for long documents
    GRAD_ACCUM_STEPS = 8  # Increased to compensate
    EPOCHS = 50 #5
    LEARNING_RATE = 1e-6

    # Binary classification labels
    
    label2id = {"O - OVERALL - NON FORMALISTIC": 0, "O - OVERALL - FORMALISTIC": 1,}
    id2label = {0: "O - OVERALL - NON FORMALISTIC", 1: "O - OVERALL - FORMALISTIC"}

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TrustHLT-ECALP/llama-3.1-8b-cpt-full")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def flatten_text_list(data):
        data['text'] = " ".join(ast.literal_eval(data['text']))
        data['f_labels'] = label2id[data['f_labels']]
        return data

    def tokenizer_function(data):
        return tokenizer(data["text"], truncation=True)

    # Load and process datasets
    dataset = load_dataset("TrustHLT/madon-init", name="default")
    dataset = dataset.map(flatten_text_list)
    tokenized_datasets = dataset.map(tokenizer_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(['text', 'labels', 'result_labels'])
    # chose the formalistic label as a target
    tokenized_datasets = tokenized_datasets.rename_column('f_labels', 'labels')
    tokenized_datasets.set_format("torch")

    # === MODEL SETUP ===
    print("\nLoading model with classification head...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        id2label=id2label, 
        label2id=label2id,
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
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
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
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
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
    logits, labels, metrics = trainer.predict(tokenized_datasets["test"])
    print(metrics)

    predictions = np.argmax(logits, axis=1)

    # Create results dataframe with UIDs
    results_df = pd.DataFrame({
        "ID": tokenized_datasets['test']["doc_id"],
        "Gold Labels": labels.tolist(),
        "Predicted Labels": predictions.tolist(),
    })

    results_df = results_df.sort_values("ID").reset_index(drop=True)

    # Add subset indicator to filename if using subset (For debugging)
    results_csv_path = os.path.join(output_dir, f"test_results_holistic_llama_{model_type}_seed_{seed}.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    print(classification_report(labels, predictions))
    
    print(f"=== Completed experiment: {exp_name} ===\n")

