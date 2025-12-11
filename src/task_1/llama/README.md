#### Overview
- This code provides a complete fine-tuning pipeline for **binary classification** using **Llama-3.1 8B** on the **MADON dataset**.  
- It classifies each paragraph as either **“Argument”** or **“No Argument.”**

- The workflow is built upon **Hugging Face Transformers**, **Comet ML** for experiment tracking, and **PyTorch** for training.  
- The code ensures reproducibility through fixed random seeds and deterministic CUDA behavior.
- This script fine-tunes a **Llama-based transformer model** to detect the presence or absence of argumentative content in paragraphs from Czech legal documents.

- The binary labels used are:

| Label | Meaning |
|--------|----------|
| 0 | No Argument |
| 1 | Argument |


---

#### Dependencies

- All the libraries used for this experiment or any other Llama related experiment are available in the **requirements.txt** file
- Also install ``flash_attn==2.7.4.post1``

---

#### Dataset

- The script automatically loads the **MADON dataset** from the Hugging Face Hub:

```python
dataset = load_dataset("TrustHLT/madon-init", name="default")
```

- Each entry represents a legal document that is split into paragraphs.
- Each paragraph is labeled as either containing an argument (`1`) or not (`0`).

---

#### Usage

##### Basic Command

```bash
python binary_full.py \
  --model meta-llama/Llama-3.1-8B \
  --seed 32 \
  --output_dir llama_standard_binary_32 \
  --training_type standard
```

##### GPU Configuration

The script automatically detects and uses CUDA when available:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

#### Arguments

| Argument          | Type  | Default                      | Description                                  |
| ----------------- | ----- | ---------------------------- | -------------------------------------------- |
| `--model`         | `str` | `"meta-llama/Llama-3.1-8B"`  | Model name or local path                     |
| `--seed`          | `int` | `32`                         | Random seed for reproducibility              |
| `--output_dir`    | `str` | `"llama_standard_binary_32"` | Directory for saving checkpoints and outputs |
| `--training_type` | `str` | `"standard"`                 | Tag to label experiment type                 |

---

#### Code Structure

```
binary_full.py
│
├── process_data()               # Converts dataset into paragraph-level binary format
├── tokenizer_function()         # Tokenizes text inputs using the model tokenizer
├── compute_metrics()            # Computes accuracy and classification report
├── Trainer()                    # Hugging Face Trainer setup
└── main()                       # Full training and evaluation flow
```

---

#### Functions and Classes

##### 1. `process_data(split, quick=False)`

- Processes raw dataset splits into paragraph-level binary-labeled examples.

- **Inputs:**

    * `split`: Dataset split (train/validation/test)
    * `quick`: Optional boolean to subsample ~3% of data for debugging

- **Returns:** Dictionary with keys:

    * `text`: list of paragraphs
    * `labels`: list of binary labels
    * `uid`: unique paragraph IDs

---

##### 2. `tokenizer_function(data)`

- Tokenizes paragraphs using the **AutoTokenizer** from the provided model.

```python
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
```

- Padding is set to the left with the end-of-sequence token as the pad token.

---

##### 3. `compute_metrics(eval_pred)`

- Computes performance metrics after each evaluation phase.

- **Metrics:**

    * `accuracy`
    * Full `classification_report` (precision, recall, F1 per class)

- The predictions are obtained by applying `argmax` to the logits.

---

##### 4. `Trainer`

- Implements Hugging Face’s **Trainer API** for fine-tuning.

- Key training configurations:

    * `gradient_checkpointing=True`
    * `optim="adamw_bnb_8bit"` (8-bit optimizer for memory efficiency)
    * `lr_scheduler_type="cosine"`
    * `warmup_ratio=0.1`
    * `bf16=True` for efficient mixed precision

- The model and tokenizer are automatically saved after each epoch.

---

#### Output Files

- All outputs are saved inside the directory specified by `--output_dir`.

| File                                                        | Description                        |
| ----------------------------------------------------------- | ---------------------------------- |
| `checkpoint-*`                                              | Model checkpoints after each epoch |
| `final_model-epoch-10/`                                     | Final trained model                |
| `test_results_binary_llama_<training_type>_seed_<seed>.csv` | Predictions and labels on test set |

- Each CSV contains:

    * `ID`: Paragraph identifier
    * `Gold Labels`: Ground truth (0 or 1)
    * `Predicted Labels`: Model predictions

---

#### Metrics

- At the end of training, the script prints:

    * Per-class **precision**, **recall**, **F1-score**
    * **Overall accuracy**
    * Full **classification report** for the test set

- The same metrics are also returned by the `Trainer` and logged to **Comet ML** (if enabled).

---

#### Notes

* To enable **Comet ML logging**, replace:

  ```python
  experiment = Experiment("Fill this with your Comet API key")
  ```

    - with your actual API key and ensure you are logged into Comet.

* Deterministic training is enforced by:

  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```

    - This guarantees reproducibility across runs.

* If you want faster debugging, set `quick=True` inside `process_data()` to train on ~3% of the dataset.


