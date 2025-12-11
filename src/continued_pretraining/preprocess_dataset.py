import tarfile
import xml.etree.ElementTree as ET
import os
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

def process(outputpath: str, tarpath: str, parquetpath: str, unzip: bool = False, rewrite: bool = False) -> Dataset:
    # unzip the files if necessary
    if unzip:
        with tarfile.open(tarpath, "r:bz2") as tar:
            tar.extractall(outputpath)
    all_docs = []
    if not os.path.exists(parquetpath) or rewrite:
        # process all .xml files in the directory
        for f in tqdm([os.path.join(outputpath, f) for f in os.listdir(outputpath) if f.endswith('.xml')]):
            root = ET.parse(f).getroot()
            doc = ''
            for par in root.findall('p'):
                partext = par.get('text')
                doc += partext + ' '
            all_docs.append({'document_id': f, 'text': doc})

        df_czech_docs = pd.DataFrame(all_docs)
        df_czech_docs.to_parquet(parquetpath, index=False)

    dataset = Dataset.from_parquet(parquetpath)
    return dataset


def process_nssoud(outputpath: str, tarpath: str, parquetpath: str, unzip: bool = False, rewrite: bool = False):
    # special handling for nssoud files as .txt
    # unzip the files if necessary
    if unzip:
        with tarfile.open(tarpath, "r:bz2") as tar:
            tar.extractall(outputpath)

    if (not os.path.exists(parquetpath)) or rewrite:
        # append "out" to output path because the dir is not flat
        #outputpath = os.path.join(outputpath, 'out')
        all_docs = []
        # process all .txt files
        for f in tqdm([os.path.join(outputpath, f) for f in os.listdir(outputpath) if f.endswith('.txt')]):
            document_name = f.split('/')[-1].split('.')[0]
            with open(f, 'r') as rf:
                doc = rf.readlines()
                doc = ' '.join(doc)
            all_docs.append({'document_id': document_name, 'text': doc})

        df_czech_docs = pd.DataFrame(all_docs)
        # direct conversion from pandas to parquet can cause OOM, so we use pyarrow directly to save
        table = pa.Table.from_pandas(df_czech_docs)
        pq.write_table(table, parquetpath)
    
    dataset = Dataset.from_parquet(parquetpath)
    return dataset


if __name__ == '__main__':

    # pretraining data obtained and copied into the directory (refer to README)
    tarpath_nsoud = 'data/pretraining_data/nsoud.tar.bz2'
    tarpath_nssoud = 'data/pretraining_data/nssoud.tar.bz2'

    # extraction paths for unzipped files (just temporary, can be removed later)
    outputpath_nsoud = 'data/pretraining_data/nsoud/xml'
    outputpath_nssoud = 'data/pretraining_data/nssoud/xml'

    # the output path of the preprocessed data
    parquetpath_nsoud = 'data/pretraining_data/nsoud/nsoud.parquet'
    parquetpath_nssoud = 'data/pretraining_data/nssoud/nssoud.parquet'


    # unzip nsoud and process
    ds_nsoud = process(outputpath_nsoud, tarpath_nsoud, parquetpath_nsoud, unzip=False, rewrite=False)
    print(ds_nsoud)

    # unzip nssoud and process
    ds_nssoud = process(outputpath_nssoud, tarpath_nssoud, parquetpath_nssoud, unzip=False, rewrite=False)
    print(ds_nssoud)

    # for convenience in pretraining scripts, push to hub
    hf_hub_name = 'yourname/madon_czech_legal'
    #ds_nsoud.push_to_hub(hf_hub_name, "NSOUD", private=True)
    #ds_nssoud.push_to_hub(hf_hub_name, "NSSOUD", private=True)
