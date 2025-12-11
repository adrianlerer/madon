import os
from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

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
    parser.add_argument('--model', type=str, default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='Llama3.1-8B-czech-legal')
    parser.add_argument('--comet_workspace', type=str, default=None)
    parser.add_argument('--comet_project', type=str, default=None)
    parser.add_argument('--hub_model_name', type=str, default=None)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    # arguments
    exp_name = args.output_dir
    model_path = args.model
    seed = args.seed
    output_dir = args.output_dir
    comet_workspace = args.comet_workspace
    comet_project = args.comet_project
    hub_model_name = args.hub_model_name
    resume = args.resume

    # other parameters
    logging = False # logging is turned on automatically if comet directories are provided
    doc_map_path = "doc_map.csv" # document needed to filter dataset
    presaved_tokenized_dataset = 'tokenized_dataset'

    # logging with cometml
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


    max_seq_length = 32_000
    dtype = None
    # we do not use the quantized version
    load_in_4bit = False
    # we finetune full without LoRA
    ft_full = True

    ds = prepare_dataset(seed=seed, remove_documents=doc_map_path).train_test_split(test_size=0.005)

    # training cpt with unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        full_finetuning = ft_full,
    )

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        # add the EOS_TOKEN to the end of each sample.
        return { "text" : [example + EOS_TOKEN for example in examples["text"]] }

    # Apply the fomatting to all of the samples in the dataset.
    ds = ds.map(formatting_prompts_func, batched = True,)

    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = ds,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 8,

        args = UnslothTrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            num_train_epochs = 1,
            learning_rate = 5e-5,
            embedding_learning_rate = 5e-6,
            lr_scheduler_type = "cosine",
            warmup_ratio = 0.1,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            optim = "adamw_8bit",
            weight_decay = 0.00,
            logging_steps = 1,
            report_to = ["comet_ml"] if logging else 'none',
            seed = seed,
            output_dir = os.path.join(output_dir, 'checkpoint'),
            save_steps = 200,
            save_total_limit= 2,
        ),
    )

    trainer_stats = trainer.train(resume_from_checkpoint=resume)

    print(trainer_stats)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # optional: push to hub
    if hub_model_name != None:
        model.push_to_hub(hub_model_name, tokenizer, private=True)
