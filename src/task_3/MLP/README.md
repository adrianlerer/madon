# Holistic formalism classification experiments with MLP
## What modules are included here?
### Configuration [configuration.py](configuration.py)
Utility class to manage experiment configuration, device selection and deterministic seeding for PyTorch-based experiments.

The configuration object mainly handles the following processes:
  - It loads and persists experiment parameters to `exp_{exp_num}/seed_{seed}/config.pickle` under `result_path`.
  - Choose compute device (CUDA, MPS on macOS, or CPU).
  - Initialize deterministic seeds for Python, NumPy and PyTorch (CUDA or MPS as applicable).

### ProcessFeatures [process_features.py](process_features.py)
The object creates data based on the document statistics for the feature based holistic classification of formalism.

The object handles following matters:
- Checks whether arguments were set correctly (e.g., split must be set to True and is_meta must be false);
- Uses raw data from INCePTION to extract document level statistics;
- Resulting dataset (*features.pickle*) can be found in the [gold](../../../data/processed_datasets/gold)
- It also helps us to specify setup for the inference configuration via ```getfname()``` function;

### FeatSet [dataset.py](dataset.py)
The object creates a dataset object that is compatible for torch's data loader and the training process;

### Model [model.py](model.py)
This the classifier object. Notice that model architecture is dynamic and heavily dependent on the architecture information you
provide in the arguments, namely ```hidden_dim_list``` and ```dropout_list```.

### Training [train.py](train.py)
Training object that handles following processes:
- Training and testing the classifier;
- Loading and evaluating the classifier based on the best validation performance;
- Running 3E pipeline;

### Significance [significance.py](significance.py)
The object handles explaining the feature importance for the classification task. We use LIME and SHAP techniques,
however we use SHAP in our reporting, as it is more reliable in the global setup (additive property).

### Best Model [exp_43](experiments/exp_43)
The exp_43 folder has all 3 seeds of the best model

---

## How to run the module separately?
### How to train the model:
- Running the following code will perform the training only:
```
python main.py --split --exp_num 1 --normalize --seed 42 --epochs 20 --train
```
- Details on parameters:
  - split: boolean to specify whether dataset must be split or not (notice: if you don't split, you can't run the MLP);
  - exp_num: experiment number that specifies the current experimental setup;
  - normalize: boolean variable which lets us normalize the dataset;
  - seed: seed value for the experimental setup;
  - epochs: how many epochs would you like to train the model?;
  - train: specifies whether training must be run.
    - Note: If chosen experiment number and seed value combination was not run before, then the new experimental results
      will be saved under the specific folder. Otherwise, it won't training will be skipped to prevent ruining saved
      outcome;

### How to test the model:
- Running the following code will perform training (if has not been done before), then test the model
```
python main.py --split --exp_num 1 --train --normalize --seed 42 --epochs 20 --train --test
```

### Significance scores:
Running any command above will run the significance scoring. However, if you do not have any trained MLP model, then you
will face with an error.
