

---

#### Overview

- This code provides for fine-tuning **Llama-3.1 8B** on the **MADON dataset** for **multi-label legal argument classification**. It uses Hugging Face Transformers, PEFT (Only for some experiments, the best models introduced in this paper were fully trained without using LoRA), and Comet ML (optional) for experiment tracking.  
- The code supports both **asymmetric loss** and **binary cross-entropy (BCE)** for multi-label optimization.

- The script fine-tunes the **Llama-3.1 8B** model (or the Llama **CPT** variant) for **multi-label paragraph-level classification** of legal arguments in Czech legal documents.  
- Each paragraph can have multiple labels chosen from eight predefined categories:

    - ["LIN", "SI", "CL", "D", "HI", "PL", "TI", "PC"]

The model uses either:
- **Asymmetric Loss** (for better handling of label imbalance), or
- **BCEWithLogitsLoss** (standard binary loss).

---

#### Dependencies


- All the libraries used for this experiment or any other Llama related experiment are available in the **requirements.txt** file
- Also install ``flash_attn==2.7.4.post1``

---

#### Dataset

The code automatically downloads and loads the dataset from the Hugging Face Hub:

```python
dataset = load_dataset("TrustHLT/madon-init", name="default")
```

Each document is split into paragraphs with associated label vectors.

---

#### Usage

##### Running the Training Script

```bash
python asy_and_bce_full.py \
  --model meta-llama/Llama-3.1-8B \
  --seed 42 \
  --output_dir llama_standard_multi_label_42 \
  --training_type standard \
  --loss asymmetric \
  --gamma 4
```

---

#### Arguments

| Argument          | Type  | Default                           | Description                                 |
| ----------------- | ----- | --------------------------------- | ------------------------------------------- |
| `--model`         | `str` | `"meta-llama/Llama-3.1-8B"`       | Model name or path (Hugging Face format)    |
| `--seed`          | `int` | `42`                              | Random seed for reproducibility             |
| `--output_dir`    | `str` | `"llama_standard_multi_label_42"` | Directory to save checkpoints and results   |
| `--training_type` | `str` | `"standard"`                      | Optional tag to distinguish experiment type |
| `--loss`          | `str` | `"asymmetric"`                    | Loss type: `"asymmetric"` or `"bce"`        |
| `--gamma`         | `int` | `4`                               | Gamma parameter for the Asymmetric Loss     |

---

#### Code Structure

```
asy_and_bce_full.py
│
├── AsymmetricLossOptimized       # Custom loss for label imbalance
├── process_data()                # Converts dataset into paragraph-level format
├── tokenize_function()           # Tokenizes text using the model tokenizer
├── MultiLabelTrainerASYM         # Trainer subclass for Asymmetric Loss
├── MultiLabelTrainerBCE          # Trainer subclass for BCE Loss
├── compute_metrics()             # Defines custom F1/Accuracy evaluation
│
└── main()                        # Complete training and evaluation flow
```

---

#### Functions and Classes

##### 1. `AsymmetricLossOptimized`

A custom implementation of the **Asymmetric Loss** function optimized for GPU memory efficiency.
This loss focuses more on hard-to-classify positive samples by applying separate focusing parameters for positive and negative labels.

**Parameters:**

* `gamma_neg`: focusing parameter for negative examples
* `gamma_pos`: focusing parameter for positive examples
* `clip`: value for probability clipping
* `eps`: small value to prevent log(0)

**Returns:**
Scalar loss value for backpropagation.

---

##### 2. `process_data(split)`

Transforms dataset entries into paragraph-level samples with corresponding binary label vectors.

**Inputs:**

* `split`: a dataset split (train/validation/test) from Hugging Face

**Returns:**
Dictionary with keys:

* `"text"`: list of paragraph strings
* `"labels"`: binary vectors (length = 8)
* `"uid"`: unique identifier per paragraph

---

##### 3. `tokenize_function(examples)`

Tokenizes text data using the provided tokenizer.
Sets `padding="max_length"` and truncates to `MAX_LENGTH`.

**Returns:** tokenized batch including `input_ids`, `attention_mask`, and `labels`.

---

##### 4. `MultiLabelTrainerASYM` and `MultiLabelTrainerBCE`

Custom subclasses of `transformers.Trainer` overriding the `compute_loss()` function:

* `MultiLabelTrainerASYM`: uses `AsymmetricLossOptimized`
* `MultiLabelTrainerBCE`: uses `torch.nn.BCEWithLogitsLoss`

Both classes support Hugging Face’s standard training arguments and evaluation flow.

---

##### 5. `compute_metrics(p)`

Computes **F1 scores**, **accuracy**, and per-label performance.
Applies a 0.5 sigmoid threshold to logits before metric calculation.

**Outputs:**

* `Per-label F1`
* `f1_micro`, `f1_macro`
* `accuracy`

---

#### Output Files

The script produces several output artifacts inside `--output_dir`:

| File                                                                | Description                                                 |
| ------------------------------------------------------------------- | ----------------------------------------------------------- |
| `checkpoint-*`                                                      | Hugging Face checkpoints per epoch                          |
| `final_model-epoch-3/`                                              | Final trained model                                         |
| `paragraph-multilabel-llama-<training_type>-<loss>-seed-<seed>.csv` | CSV containing paragraph UIDs, gold labels, and predictions |

Each CSV entry includes:

* `UID`: paragraph identifier
* `Gold Labels`: ground truth vectors
* `Predicted Labels`: model predictions

---

#### Metrics

At the end of training, the script prints:

* Per-label classification reports (`absent` vs `present`)
* Overall **Micro-F1**, **Macro-F1**, and **Accuracy** scores on the test set.

---

#### Notes

* The `comet_ml` experiment section is included but commented out.
  Uncomment and configure it with your workspace/project name for online experiment tracking.

* To use a **subset of the dataset** for quick testing, set:

  ```python
  USE_SUBSET = True
  ```
