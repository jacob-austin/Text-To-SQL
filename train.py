import pickle
import random
import argparse
from functools import partial
import os
import logging
import math
import utils
from packaging import version

import torch

import datasets
import pandas as pd 

import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from process_sql import tokenize

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoModelWithLMHead



logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train machine translation transformer model")


    #required argument
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     required=True,
    #     help=("Where to store the final model. "
    #           "Should contain the source and target tokenizers in the following format: "
    #           r"output_dir/{source_lang}_tokenizer and output_dir/{target_lang}_tokenizer. "
    #           "Both of these should be directories containing tokenizer.json files."
    #     ),
    # )

    #optional
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="stas/wmt14-en-de-pre-processed",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="en-de",
        help=("Many datasets in Huggingface Dataset repository have multiple versions or configs. "
              "For the case of machine translation these usually indicate the language pair like "
              "en-es or zh-fr or similar. To look up possible configs of a dataset, "
              "find it on huggingface.co/datasets."),
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Whether to use a small subset of the dataset for debugging.",
    )
    # Model arguments
    parser.add_argument(
        "--num_layers",
        default=6,
        type=int,
        help="Number of hidden layers in the Transformer encoder",
    )
    parser.add_argument(
        "--hidden_size",
        default=512,
        type=int,
        help="Hidden size of the Transformer encoder",
    )
    parser.add_argument(
        "--num_heads",
        default=8,
        type=int,
        help="Number of attention heads in the Transformer encoder",
    )
    parser.add_argument(
        "--fcn_hidden",
        default=2048,
        type=int,
        help="Hidden size of the FCN",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="The maximum total sequence length for source and target texts after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=None,
        help="Overwrite the cached training and evaluation sets",
    )

    # Training arguments
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu) on which the code should run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout rate of the Transformer encoder",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=5000,
        help="Perform evaluation every n network updates.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Compute and log training batch metrics every n steps.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=transformers.SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--generation_type",
        choices=["greedy", "beam_search"],
        default="beam_search",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help=("Beam size for beam search generation. "
              "Decreasing this parameter will make evaluation much faster, "
              "increasing this (until a certain value) would likely improve your results."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--wandb_project", 
        default="transformer_mt",
        help="wandb project name to log metrics to"
    )



    args = parser.parse_args()

    return args

def preprocess_function(examples, source_tokenizer, target_tokenizer, max_seq_length):
    

    inputs = examples['question']
    targets = examples['query']

    model_inputs = source_tokenizer(inputs, max_length=max_seq_length, truncation=True)
    
    target_ids = target_tokenizer(targets, max_length=max_seq_length - 1, truncation=True)
    
    decoder_input_ids = []
    labels = []

    print(model_inputs['input_ids'][0])
    print(model_inputs['input_ids'][1])


    for target in target_ids:
        decoder_input_ids.append(target)
        labels.append(target)

    model_inputs["decoder_input_ids"] = decoder_input_ids
    model_inputs["labels"] = labels

    return model_inputs

def collation_function_for_seq2seq(batch, source_pad_token_id, target_pad_token_id):
    """
    Args:
        batch: a list of dicts of numpy arrays with keys
            input_ids
            decoder_input_ids
            labels
    """
    input_ids_list = [ex["input_ids"] for ex in batch]
    decoder_input_ids_list = [ex["decoder_input_ids"] for ex in batch]
    labels_list = [ex["labels"] for ex in batch]

    collated_batch = {
        "input_ids": utils.pad(input_ids_list, source_pad_token_id),
        "decoder_input_ids": utils.pad(decoder_input_ids_list, target_pad_token_id),
        "labels": utils.pad(labels_list, target_pad_token_id),
    }

    collated_batch["encoder_padding_mask"] = collated_batch["input_ids"] == source_pad_token_id
    return collated_batch


sql_vocab = [
    'select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'by', 'having', 'distinct',
    'join', 'on', 'as', 'outer', 'inner',
    'not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists',
    'none', 'max', 'min', 'count', 'sum', 'avg',
    'and', 'or',
    'intersect', 'union', 'except',
    'desc', 'asc',
    ',', '(', ')', ';', '*'
]



BOS_INDEX = 0
UNK_INDEX = 1
EOS_INDEX = len(sql_vocab) + 2
PAD_INDEX = len(sql_vocab) + 3

def build_sql_id_vector(str):
    tokens = tokenize(str.lower())

    ids = []
    for token in tokens:
        if token in sql_vocab:
            ids.append(sql_vocab.index(token) + 2)
        else:
            ids.append(UNK_INDEX)
    return ids

def sql_tokenizer(targets):
    sql_ids = []

    for ex in targets:
        sql_ids.append(build_sql_id_vector(ex))
    
    return sql_ids

def sql_decoder(ids):
    tokens = []

    for id in ids:
        if id == BOS_INDEX:
            tokens.append('[BOS]')
        elif id == UNK_INDEX:
            tokens.append('[UNK]')
        elif id == EOS_INDEX:
            tokens.append('[EOS]')
        elif id == PAD_INDEX:
            tokens.append('[PAD]')
        else:
            tokens.append(sql_vocab[id - 2])
    
    return tokens

def main():
    args = parse_args()
    logger.info(f"Starting script with arguments: {args}")

    # dataset = None

    # with open("dataset/classical_test.pkl", "rb") as file:
    #     dataset = pickle.load(file)

    dataset = load_dataset('spider')

    # dataset_pd = pd.DataFrame(dataset)
    # dataset_pd = dataset_pd[['text', 'query', 'variables', 'db_path']]

    # raw_dataset = datasets.Dataset.from_pandas(dataset_pd)
    # raw_datasets = datasets.DatasetDict({"train": raw_dataset })
    # raw_datasets = raw_datasets['train'].train_test_split(test_size=int(len(raw_dataset) * .2),)



    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

    model.to(args.device)

    column_names = dataset["train"].column_names

    preprocess_function_wrapped = partial(
        preprocess_function,
        max_seq_length=args.max_seq_length,
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
    )

    processed_datasets = dataset.map(
        preprocess_function_wrapped,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
   

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"] if "validation" in processed_datasets else processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 2):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        logger.info(f"Decoded input_ids: {tokenizer.decode(train_dataset[index]['input_ids'])}")
        logger.info(f"Decoded labels: {sql_decoder(train_dataset[index]['labels'])}")
        logger.info("\n")

    # collation_function_for_seq2seq_wrapped = partial(
    #     collation_function_for_seq2seq,
    #     source_pad_token_id=tokenizer.pad_token_id,
    #     target_pad_token_id=tokenizer.pad_token_id,
    # )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collation_function_for_seq2seq_wrapped, batch_size=args.batch_size
    )

    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=collation_function_for_seq2seq_wrapped, batch_size=args.batch_size
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = transformers.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps))


    # Log a pre-processed training example to make sure the pre-processing does not have bugs in it
    # and we do not input garbage to our model.
    batch = next(iter(train_dataloader))
    logger.info("Look at the data that we input into the model, check that it looks like what we expect.")
    for index in random.sample(range(len(batch)), 2):
        logger.info(f"Decoded input_ids: {tokenizer.decode(batch['input_ids'][index])}")
        logger.info(f"Decoded labels: {sql_decoder(batch['labels'][index])}")
        logger.info("\n")

if __name__ == "__main__":
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")

    main()
