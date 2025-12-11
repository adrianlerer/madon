from datasets import load_dataset, concatenate_datasets, Dataset
import os
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "1"

PRETRAINING_DATAPATH = "TrustHLT/czech-legal-NSSOUD-NSOUD"

def prepare_dataset(remove_documents: str, seed: int = 42) -> Dataset:
    # download datasets and concatenate them into a single dataset (datasets are formatted the same)
    ds_nsoud = load_dataset(PRETRAINING_DATAPATH, 'NSOUD')
    ds_nssoud = load_dataset(PRETRAINING_DATAPATH, 'NSSOUD')
    ds = concatenate_datasets([ds_nsoud['train'], ds_nssoud['train']])

    # load document containing ids of decisions that are already contained 
    # in the annotated downstream task dataset -> for removal
    df_doc_ids = pd.read_csv(remove_documents)
    df_doc_ids['original_id'] = df_doc_ids['original_id'].apply(lambda x: x.replace('.html', '.xmi'))
    df_doc_ids['original_id'] = df_doc_ids['original_id'].apply(lambda x: x.split('.xmi')[0])

    # filter out documents that appear in the annotated madon dataset
    ds = ds.filter(lambda s: not(s["document_id"].split('/')[-1].split('.xml')[0] in df_doc_ids['original_id'].values))
    ds = ds.shuffle(seed=seed)
    return ds

def train_tokenizer_from_pretrained(dataset_iterator) -> None:
    from transformers import AutoTokenizer

    old_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

    # sample to compare different tokenization
    sample = 'Předmětem sporu je otázka, zda má být daňová uznatelnost úroků z dluhopisů vyplacených v roce 2013 posuzována podle § 24 odst. 2 písm. zi) zákona o daních z příjmů, ve znění účinném od 1. 1. 2005 do 31. 12. 2013, nebo podle generální klausule v § 24 odst. 1 téhož zákona.'
    print(old_tokenizer.tokenize(sample))

    # train new tokenizer from ModernBERT tokenizer using the same config
    tokenizer = old_tokenizer.train_new_from_iterator(dataset_iterator, vocab_size=old_tokenizer.vocab_size, length=300_000)

    print(tokenizer.tokenize(sample))
    tokenizer.save_pretrained('czech_legal_modernbert_tokenizer')

if __name__ == "__main__":

    # prepare dataset and filter documents which are part of the annotated gold data
    doc_map_path = 'doc_map.csv'
    dataset = prepare_dataset(remove_documents=doc_map_path)

    print("Training tokenizer from ModernBERT config")

    def batch_iterator(dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    train_tokenizer_from_pretrained(batch_iterator(dataset, 1000))

    print("Tokenizer training complete.")