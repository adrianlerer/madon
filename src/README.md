Here are all the source files. Please consider, that the MADON dataset should be uploaded on HuggingFace to work with these scripts. The scripts load the dataset from HuggingFace, preprocess it, finetune the model on that and predict the labels based on the finetuned models.

The naming convention is the same used in the paper:

- Task 1: Argument presence detection (Paragraph Binary)
- Task 2: Argument type classification (Paragraph Multilabel)
- Task 3: Holistic formalism classification (Overall Binary)

You will find a general evaluation folder, in which the script for evaluation with the instruction is located. Additionally, there is an evaluation folder for each task, which contains the evaluation data (not script) of all models for that specific task.

