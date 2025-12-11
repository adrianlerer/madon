This is the public repository for the paper "Mining Legal Arguments to Study Judicial Formalism".

# Models
The models are available on Huggingface. Only the best models are uploaded:

Task 1 (Argument presence detection): ModernBERT-large-Czech-Legal: https://huggingface.co/TrustHLT/ModernBERT-large-madon-arg-detection

Task 2 (Argument type classification): Llama 3.1 8B (Full) Base + Asy: https://huggingface.co/TrustHLT/Llama-3.1-8B-madon-arg-classification

Task 3 (Holistic formalism classification): ModernBERT-large-Czech-Legal: https://huggingface.co/TrustHLT/ModernBERT-large-madon-formalism

# Data
- In the `./data` folder you will find:
	- Information about the data
	- Annotation details
	- Links to the datasets
	- Code for processing the data

# Src
- Here, you will find all codes for:
	- Processing the data for finetuning the Models
	- Finetuning
	- Prediction
	- Evaluation
- each for all respective models, that are relevant as in the paper.

