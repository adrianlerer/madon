## Continued Pretraining scripts
This directory contains scripts that were used to continue pretraining the models ModernBERT-large and Llama 3.1 8B base.
The following steps have to be done to reproduce the pretraining data preprocessing and training:
1. Obtain the raw pretraining data from https://git-ce.rwth-aachen.de/trusthlt-public/czech-supreme-courts-data-and-crawler
   a) Download the nsoud xml files
   ```bash
   curl -L -o data/pretraining_data/nsoud.tar.bz2 "https://git-ce.rwth-aachen.de/trusthlt-public/czech-supreme-courts-data-and-crawler/-/archive/main/czech-supreme-courts-data-and-crawler-main.tar.bz2?path=data/nsoud/nsoud-xml"
   ```
   Extract the files, concatenate the chunks and extract into the correct subfolder (nsoud/xml)
   ```bash
   tar -xjf data/pretraining_data/nsoud.tar.bz2 --strip-components=3 -C data/pretraining_data/ czech-supreme-courts-data-and-crawler-main-data-nsoud-nsoud-xml/data/nsoud
   cat data/pretraining_data/nsoud-xml/nsoud-scraped-extracted-xml-as-of-2024-09-13.tar.gz.* > data/pretraining_data/nsoud.tar.gz
   mkdir -p data/pretraining_data/nsoud/ && tar -xzvf data/pretraining_data/nsoud.tar.gz -C data/pretraining_data/nsoud/
   ```
   Delete temporary files and folders
   ```bash
   rm -f data/pretraining_data/nsoud.tar.gz && rm -rf data/pretraining_data/nsoud-xml
   rm -f data/pretraining_data/nsoud.tar.bz2
   ```
   b) Download the nssoud xml files
   ```bash
   curl -L -o data/pretraining_data/nssoud.tar.bz2 "https://git-ce.rwth-aachen.de/trusthlt-public/czech-supreme-courts-data-and-crawler/-/archive/main/czech-supreme-courts-data-and-crawler-main.tar.bz2?path=data/nssoud/nssoud-xml"
   ```
   Extract the files, concatenate the chunks and extract into the correct subfolder (nssoud/xml)
   ```bash
   tar -xjf data/pretraining_data/nssoud.tar.bz2 --strip-components=3 -C data/pretraining_data/ czech-supreme-courts-data-and-crawler-main-data-nssoud-nssoud-xml/data/nssoud/
   cat data/pretraining_data/nssoud-xml/nssoud-scraped-xml-as-of-2024-11-29.tar.gz.* > data/pretraining_data/nssoud.tar.gz
   mkdir -p data/pretraining_data/nssoud/xml && tar -xzvf data/pretraining_data/nssoud.tar.gz --strip-components=1 -C data/pretraining_data/nssoud/xml nssoud-scrape-2024-11-29-xml/
   ```
   Delete temporary files and folders
   ```bash
   rm -f data/pretraining_data/nssoud.tar.gz && rm -rf data/pretraining_data/nssoud-xml
   rm -f data/pretraining_data/nssoud.tar.bz2
   ```
   
3. Set the correct directories and huggingface hub name (if needed) in [preprocess_dataset.py](preprocess_dataset.py)
4. (optional) If needed install the requirements.txt in your environment, or use either of the environments used in Llama3.1 or ModernBERT
5. Run the preprocessing step, this should produce a pretraining dataset as a hf repo
6. Go to the directory [Llama3.1](Llama3.1) or [ModernBERT](ModernBERT) to run pretraining
