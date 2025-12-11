#### Overview

- This script fine-tunes a **Llama 3.1 8B** model for **binary text classification** on the *MADON* legal dataset. The task focuses on distinguishing between **formalistic** and **non-formalistic** overall reasoning styles in legal texts.

- The process includes data preprocessing, tokenization, model training, evaluation, and result logging. The code leverages Hugging Face’s `transformers`, `datasets`, and `Trainer` API for efficient training and evaluation on GPU hardware.

---

#### Parameters

- The script accepts the following arguments:

| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--model` | `str` | `meta-llama/Llama-3.1-8B` | Path or identifier of the base model to fine-tune. |
| `--seed` | `int` | `42` | Random seed for reproducibility. |
| `--output_dir` | `str` | `llama_standard_multi_label_42` | Directory for saving models and results. |
| `--training_type` | `str` | `standard` | Descriptor for the training configuration (affects output naming). |

---

#### Process Description

##### 1. **Environment Setup**
- All random number generators (`torch`, `numpy`, `random`) are seeded to ensure reproducibility. Deterministic GPU operations are enforced by setting:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

- Additionally, environment variables are configured for deterministic CUDA and FlashAttention execution.

---

##### 2. **Dataset Preparation**

- The script loads the **MADON dataset** (`TrustHLT/madon-init`) using the Hugging Face `datasets` library.

- Each document’s text is represented as a list of strings and is flattened into a single string:

```python
data['text'] = " ".join(ast.literal_eval(data['text']))
```

- The label `f_labels` (formalistic/non-formalistic) is mapped to numerical values according to:

```python
label2id = {
    "O - OVERALL - NON FORMALISTIC": 0,
    "O - OVERALL - FORMALISTIC": 1,
}
```

---

##### 3. **Tokenization**

- The Llama tokenizer (`TrustHLT-ECALP/llama-3.1-8b-cpt-full`) processes text with truncation enabled and left-side padding.
- The padding token is set to the model’s end-of-sequence (EOS) token for compatibility with LLaMA architectures.

---

##### 4. **Model Initialization**

- A sequence classification model is loaded using:

```python
AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
```

- This setup uses bfloat16 precision and FlashAttention 2 for optimized attention computation.

---

##### 5. **Training Configuration**

- Training parameters are defined via `TrainingArguments`:

    * **Batch size:** 1 (due to document length)
    * **Gradient accumulation:** 8 steps
    * **Learning rate:** 1e-6
    * **Epochs:** 50
    * **Scheduler:** Cosine learning rate schedule with warmup ratio 0.1
    * **Optimizer:** `adamw_bnb_8bit` (efficient mixed-precision optimizer)
    * **Evaluation and saving:** Conducted at the end of each epoch

---

##### 6. **Metrics**

- The evaluation metrics include:

    * **Accuracy**
    * **Detailed classification report** (precision, recall, F1-score per class)

These are computed via:

```python
classification_report(labels, predictions, output_dict=True, zero_division=0)
```

---

##### 7. **Training and Evaluation**

- The Hugging Face `Trainer` handles training and validation:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
```

- After training, the model is saved with the final epoch number appended to the directory name.

---

##### 8. **Prediction and Results**

- Predictions on the test set are generated and saved to a CSV file:

```python
test_results_holistic_llama_{training_type}_seed_{seed}.csv
```

- Each record includes:

    * Document ID (`doc_id`)
    * Gold label
    * Predicted label

- The script also outputs a textual classification report to the console.

---

#### Output

* **Trained model directory:**
  `./{output_dir}/final_model-epoch-{EPOCHS}`
* **Evaluation results:**
  CSV file with IDs, gold labels, and predictions.
* **Console outputs:**
  Training logs, evaluation metrics, and final classification report.

---

#### Example Command

```bash
python holy_full.py \
    --model meta-llama/Llama-3.1-8B \
    --seed 42 \
    --output_dir llama_binary_finetune_42 \
    --training_type standard
```

---

#### Dependencies

- All the libraries used for this experiment or any other Llama related experiment are available in the **requirements.txt** file
- Also install ``flash_attn==2.7.4.post1``

