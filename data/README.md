	# Data folder:
## Dataset oriented information
### **Annotation scheme and guidelines**: 
- Annotation scheme and instructions for annotating decisions of the Czech Supreme Court and Supreme Administrative Court. The document describes eight legal argument types and a holistic formalism label (formalistic/non-formalistic), with definitions, examples, and flowcharts.
### **MADON dataset**: 
- Expert-annotated corpus of 272 Czech Supreme Court (1997-2024) and Supreme Administrative Court decisions (2003-2024). Paragraphs are labeled for one or more of eight argument types and each decision has a holistic formalism label. The dataset contains 1,913 annotated arguments. It was used to fine-tune and evaluate the models.
- The MADON dataset can be found here: https://git-ce.rwth-aachen.de/trusthlt-public/madon-annotated-data
### **Pretraining Corpus**: 
- The corpus of 300k unlabeled court decisions used for pretraining can be found here: https://git-ce.rwth-aachen.de/ivan.habernal.rub/czech-supreme-courts-data-and-crawler
---
## Processing the dataset for all tasks:
- This is for the processing of the MADON dataset;
- You need to preserve the same structure that is given in the link that is provided above. So just copy paste.

### When you have the raw dataset:
- If you have raw dataset put it to the [dataset/raw_data](dataset/raw_data) folder. Notice you need to have the data in two folders, namely annotated and curated:
  - annotated data is the raw data that includes all annotations by legal experts;
  - curated data is the raw data that includes the result of the legal expert annotation (i.e., curated)
  if you have this data at hand, then you can create whole dataset from scratch using the following command:
  ```
    python main.py --read_data --split
  ```
### When you do not have the raw dataset but descriptive meta dataset:
- If you do not have access to the raw data folder, we might have shared the descriptive raw data with you, which is 00-descriptive-dataset-raw.pickle:
  - You need to put this dataset into [dataset/processed_datasets](dataset/processed_datasets);
  - You can create the dataset by using this very data, simply by running the following command:
  ```
    python main.py --split
  ```
### A little more information on the dataset configurations
- Running one of the commands are given above, you can create the MADON dataset. However, notice that MADON project provides two datasets:
  - meta: where 15 legal arguments were annotated;
    - It is not a default setup. If you want to have this dataset, simply add ```--is_meta``` to the one of the commands are given above, depending on the dataset you are provided with.
  - gold: where 15 legal arguments were mapped into 8 arguments.
    - Since the paper reports the experimental results based on this dataset, we call it gold dataset.
---

## What do these objects do?
### ReadData ([read_data](read_data.py))
- Object collects all data paths in a document and saves it for easy access;
- Each sample in the saved data (it is just collected paths) includes following information:
  - data_id: document id;
  - curated_data: path to the curated data for this specific sample;
  - raw_data: path to the raw data, that includes feature information;
  - annotated_data: path to the annotations (separate from curated ones, so we put them together);
### ProcessData ([process_raw_data](process_raw_data.py))
- Object processes all collected raw files through the following steps:
  - First, tokens and paragraphs were extracted from raw data, where each is a dictionary:
    - 'id': id of the specific token or paragraph in raw data;
    - 'begin': beginning of the object (character id in the text);
    - 'end': end of the object (character id in the text);
    - 'text': textual content of the specific token or paragraph;
  - Then we collect arguments and holistic labels, where each of them is a dictionary:
    - 'begin': where the span begins;
    - 'end': where the span ends;
    - 'label_ui': User Interface label (i.e., category, since each category also has some subcategories);
    - 'label': The selected specific label (i.e., subcategory or more specific label code)
    - Notice: We collect tokenized arguments, not the wall of text;
  - We map document names to indexes and save this map;
  - Additionally, we save the list of labels just in case for easy access;
- After all these processes, we save the dataset (notice no splits here) to the '00-descriptive-dataset-meta.pickle' of the result path you provided;

- Note: No randomness in this process, thus all can be easily reproduced;
### CreateDataset ([create_dataset](create_dataset.py))
After all processes we create the dataset here through the following steps
- We collect document ids and document names, because we use document names as ids in the splits;
- We collect labels after mapping them to merged version (raw data has 15 but gold data has 8 categories) *;
- We also collect argument information with 'argument_info' key;
- We collect text as a list of paragraphs, where each paragraph is a list of tokens of the content;
- We also collect tokenized version of the whole documents (i.e., without collecting them under paragraphs)
- We also separate result labels (RRESULT) from arguments, because it is not considered as an argument;
- Notice these processes were performed for "not-split" data and we save this collected data to 01-dataset-gold-descriptive.csv;
- Notice if we choose meta above as True, the labels will not be mapped in * and we will create a dataset with 15 arguments;
- For split, we use only following features:
  - doc_id: but this time document name is used for this purpose;
  - labels: argument categories;
  - result_labels: RRESULT categories;
  - text: list of paragraphs, where each paragraph is a list of tokens;
  - f_labels: holistic labels
### DataStats ([data_statistics](data_statistics.py))
- Several statistical analysis over the dataset is done here
  - We compute Cohen's kappa, using annotations were collected with reader object;
  - We create histograms for number of arguments and tokens per document;
  - We also create argument distributions over paragraphs and documents;
