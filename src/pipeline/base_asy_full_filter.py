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
from scipy.special import expit
import pandas as pd
import random
from argparse import ArgumentParser

class AsymmetricLossOptimized(torch.nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='llama_base_asy_32')
    parser.add_argument('--training_type', type=str, default='base')
    parser.add_argument('--loss', type=str, default='asymmetric')
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--test_dir', type=str, default='mBertResults')

    args = parser.parse_args()

    exp_name = args.output_dir
    model_path = args.model
    seed = args.seed
    output_dir = args.output_dir
    model_type = args.training_type
    loss_type = args.loss
    neg_gamma = args.gamma
    test_dir = args.test_dir

    # Global Config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    EPOCHS = 5
    MAX_LENGTH = 4000
    LEARNING_RATE = 2e-5
    PENALTY_WEIGHT = 2.0
    ARGUMENT_LABELS = ["LIN", "SI", "CL", "D", "HI", "PL", "TI", "PC"]
    label2id = {'LIN': 0, 'SI': 1, 'CL': 2, 'D': 3, 'HI': 4, 'PL': 5, 'TI': 6, 'PC': 7}
    id2label = {0: 'LIN', 1: 'SI', 2: 'CL', 3: 'D', 4: 'HI', 5: 'PL', 6: 'TI', 7: 'PC'}
    USE_SUBSET = False  # Set to True to use only 1/20th of the data

    # Data preparation
    print("Loading dataset...")
    dataset = load_dataset("TrustHLT/madon-init", name="default")

    def process_data(split):
        all_texts = []
        all_labels = []
        all_ids = []

        for example in split:
            doc_id = example['doc_id']
            paragraphs = ast.literal_eval(example['text'])
            label_list = example['labels']

            for i, para in enumerate(paragraphs):
                if len(label_list[i]) == 0:
                    continue
                labels = [1 if lab in label_list[i] else 0 for lab in label2id.keys()]
                uid = f"{doc_id}_{i}"

                all_texts.append(para)
                all_labels.append(labels)
                all_ids.append(uid)

        return {"text": all_texts, "labels": all_labels, "uid": all_ids}

    def process_filtered_test_data(split, test_dir, seed):
        """
        Filter test data based on CSV file predictions.
        Only keeps examples where the predicted label is 1.
        """
        # Load the CSV filter file
        csv_filename = f"paragraph-binary-modernbert-base-seed-{seed}.csv"
        csv_path = os.path.join(test_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"Warning: Filter CSV file not found at {csv_path}")
            print("Using all test examples instead.")
            return process_data(split)
        
        # Load filter CSV
        filter_df = pd.read_csv(csv_path)
        
        # Get UIDs where predicted label is 1
        valid_uids = set()
        for _, row in filter_df.iterrows():
            predicted_label = row['Predicted Labels']
            if predicted_label == 1:
                valid_uids.add(row['UID'])
        
        print(f"Loaded filter CSV with {len(filter_df)} examples")
        print(f"Found {len(valid_uids)} examples with positive prediction")
        
        # Process test data and filter based on valid UIDs
        all_texts = []
        all_labels = []
        all_ids = []

        for example in split:
            doc_id = example['doc_id']
            paragraphs = ast.literal_eval(example['text'])
            label_list = example['labels']

            for i, para in enumerate(paragraphs):
                uid = f"{doc_id}_{i}"
                
                # Only include if UID is in valid_uids
                if uid in valid_uids:
                    labels = [1 if lab in label_list[i] else 0 for lab in label2id.keys()]
                    all_texts.append(para)
                    all_labels.append(labels)
                    all_ids.append(uid)

        print(f"Filtered test dataset: {len(all_texts)} examples (from original test set)")
        return {"text": all_texts, "labels": all_labels, "uid": all_ids}

    test_data = Dataset.from_dict(process_filtered_test_data(dataset['test'], test_dir, seed))
    dev_data = Dataset.from_dict(process_data(dataset['validation']))
    train_data = Dataset.from_dict(process_data(dataset['train']))

    dataset = DatasetDict()
    dataset['train'] = train_data
    dataset['dev'] = dev_data
    dataset['test'] = test_data


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
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    tokenized_datasets.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
    
    # === Trainer using BCE with class weights ===
    class MultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels").float()
            outputs = model(**inputs)
            logits = outputs.logits

            loss_fct = AsymmetricLossOptimized(gamma_neg=neg_gamma, gamma_pos=1, clip=0.05)
            loss = loss_fct(logits, labels)

            return (loss, outputs) if return_outputs else loss


    training_args = TrainingArguments(
        max_grad_norm=1.0,
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="steps",
        eval_steps=400,
        save_strategy="epoch",
        logging_steps=10,
        bf16=True,
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="adamw_torch",
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
        label_names=["labels"],
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
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['dev'],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    final_model_path = os.path.join(output_dir, f"final_model-epoch-{EPOCHS}")
    trainer.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")

    print("\nEvaluating on test set...")
    logits, labels, metrics = trainer.predict(tokenized_datasets["test"])
    print(metrics)

    threshold = 0.5
    predictions = (expit(logits) >= threshold).astype(int)

    # Create results dataframe with UIDs
    results_df = pd.DataFrame({
        "UID": tokenized_datasets['test']["uid"],
        "Gold Labels": labels.tolist(),
        "Predicted Labels": predictions.tolist(),
        #"Prediction_Probabilities": pred_probs.numpy()
    })

    results_df = results_df.sort_values("UID").reset_index(drop=True)

    # Add subset indicator to filename if using subset (For debugging)
    results_csv_path = os.path.join(output_dir, f"multilabel_llama_{model_type}_asy_filtered_seed_{seed}.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    # Log classification reports (for quick reference)
    print("\nClassification Reports:")
    for label in ARGUMENT_LABELS:
        idx = ARGUMENT_LABELS.index(label)
        report = classification_report(
            labels[:, idx],
            predictions[:, idx],
            target_names=["absent", "present"],
            output_dict=True
        )
        print(f"\n{label}:")
        print(classification_report(
            labels[:, idx],
            predictions[:, idx],
            target_names=["absent", "present"]
        ))
        
    # Log overall metrics (For quick reference)
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    
    print("\nOverall Metrics:")
    print(f"Micro F1: {micro_f1}")
    print(f"Macro F1: {macro_f1}")
    print(f"Accuracy: {accuracy}")
    
    print(f"=== Completed experiment: {exp_name} ===\n")

