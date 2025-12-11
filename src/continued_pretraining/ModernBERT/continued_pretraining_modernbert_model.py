from accelerate import Accelerator
import os
from transformers import (
    AutoTokenizer,
    AutoConfig, 
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    )
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
import pandas as pd
from argparse import ArgumentParser
import flash_attn

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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='answerdotai/ModernBERT-large')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='ModernBERT-large-czech-legal')
    parser.add_argument('--tokenizer_path', type=str, default='czech_legal_modernbert_tokenizer')
    parser.add_argument('--comet_workspace', type=str, default=None)
    parser.add_argument('--comet_project', type=str, default=None)
    parser.add_argument('--hub_model_name', type=str, default=None)

    args = parser.parse_args()

    # arguments
    exp_name = args.output_dir
    model_path = args.model
    seed = args.seed
    output_dir = args.output_dir
    tokenizer_path = args.tokenizer_path
    comet_workspace = args.comet_workspace
    comet_project = args.comet_project
    hub_model_name = args.hub_model_name

    # other parameters
    logging = False # logging is turned on automatically if comet directories are provided
    doc_map_path = "doc_map.csv" # document needed to filter dataset
    presaved_tokenized_dataset = 'tokenized_dataset'


    # load accelerate
    accelerator = Accelerator()
    device = accelerator.device

    # (optional) logging with cometml
    if (comet_workspace != None) & (comet_project != None):
        import comet_ml
        logging = True
        comet_ml.start(
            workspace=comet_workspace,
            project_name=comet_project,
            experiment_config=comet_ml.ExperimentConfig(
                name=exp_name,
            ),
        )

    # load dataset and prepare
    ds = prepare_dataset(seed=seed, remove_documents=doc_map_path).train_test_split(test_size=0.005)

    # load custom pretrained tokenizer
    custom_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"tokenizer is fast: {custom_tokenizer.is_fast}")

    def tokenization(example):
        return custom_tokenizer(example["text"])

    # tokenize dataset and create subsets for phases depending on token size
    # if dataset has been tokenized before, load the saved set (to save some time)
    try:
        tokenized_ds = load_from_disk(presaved_tokenized_dataset)
    except FileNotFoundError as e:
        print(f"No tokenized dataset file found: {e}")
        tokenized_ds = ds.map(tokenization, batched=True)
        tokenized_ds['train_phase_1'] = tokenized_ds['train'].filter(lambda x: len(x['input_ids']) < 8192)
        tokenized_ds['train_phase_2'] = tokenized_ds['train'].filter(lambda x: 8192 < len(x['input_ids']) < 32_000)
        #tokenized_ds['train_phase_3'] = tokenized_ds['train'].filter(lambda x: 32_000 < len(x['input_ids']) < 128_000)
        tokenized_ds.save_to_disk(presaved_tokenized_dataset)

    # configuration for checkpoints, training epochs, batch size and MLM probability for each training phase
    # this config can run on 1x A100
    phase_config = {
        'phase_1': {
            'epochs': 1,
            'batch_size': 2,
            'gac': 16,
            'mlm_prob': 0.3,
            'train_set': 'train_phase_1',
            'checkpoint_in': model_path,
            'checkpoint_out': 'cpt_modernbert_czech_legal_phase_1',
        },
        'phase_2': {
            'epochs': 1,
            'batch_size': 1,
            'gac': 8,
            'mlm_prob': 0.15,
            'train_set': 'train_phase_2',
            'checkpoint_in': 'cpt_modernbert_czech_legal_phase_1',
            'checkpoint_out': 'cpt_modernbert_czech_legal_phase_2',
        }
    }


    # train each phase, check if phase was already trained and resume from checkpoint if avail
    trained = False
    for phase, c in phase_config.items():
        if os.path.exists(c['checkpoint_out']):
            print("Phase already trained. Next phase starting...")
            continue
        else:
            print(f"Starting training for {phase}...")
            trained = True
            model_checkpoint = c['checkpoint_in']
            mlm_prob = c['mlm_prob']
            gac = c['gac']
            batch_size = c['batch_size']
            n_epochs = c['epochs']
            training_subset_name = c['train_set']
            output_path = c['checkpoint_out']

        # temporary directory for checkpoints
        tmp_dir = f'czech_legal_modernbert_{phase}'

        # resume from checkpoint if a directory exists
        resume = False
        if os.path.exists(tmp_dir):
            if len(os.listdir(tmp_dir)) != 0:
                resume = True

        model = AutoModelForMaskedLM.from_pretrained(
                    model_checkpoint,
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                ).to(device)

        config = AutoConfig.from_pretrained(
            model_checkpoint
        )
        ## potential changes to the config here

        model = AutoModelForMaskedLM.from_pretrained(
            model_checkpoint, config=config
        )

        model.to(device)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=custom_tokenizer, mlm=True, mlm_probability=mlm_prob
        )

        training_args = TrainingArguments(
            output_dir=tmp_dir,          # output directory to where save model checkpoint
            eval_strategy="steps",    # evaluate each `logging_steps` steps     
            num_train_epochs=n_epochs,            # number of training epochs, feel free to tweak
            per_device_train_batch_size=batch_size, # the training batch size, put it as high as your GPU memory fits
            gradient_accumulation_steps=gac,  # accumulating the gradients before updating the weights
            per_device_eval_batch_size=1,  # evaluation batch size
            logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
            save_steps=1000,
            # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
            save_total_limit=2,           # whether you don't have much space so you let only 3 model weights saved in the disk
        )


        trainer = accelerator.prepare(Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds[training_subset_name],
            eval_dataset=tokenized_ds["test"],
            data_collator=data_collator,
            processing_class=custom_tokenizer,
        ))

        trainer.train(resume_from_checkpoint=resume)

        accelerator.wait_for_everyone()

        # save model and tokenizer
        model.save_pretrained(output_path)
        custom_tokenizer.save_pretrained(output_path)

    # if everything was already trained, reload model for saving
    if not trained:
        model = AutoModelForMaskedLM.from_pretrained(
                        model_checkpoint,
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True,
                        attn_implementation="flash_attention_2",
                        torch_dtype=torch.bfloat16,
                    ).to(device)

    # save model after final phase to output directory
    model.save_pretrained(output_dir)
    custom_tokenizer.save_pretrained(output_dir)
    # optional: push to hub
    if hub_model_name != None:
        model.push_to_hub(hub_model_name, custom_tokenizer, private=True)