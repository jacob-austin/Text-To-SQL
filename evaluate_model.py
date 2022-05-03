import datasets
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from datasets import Dataset, load_dataset
from functools import partial

import random
import os
import logging
import tqdm
from tqdm.auto import tqdm

import logging

import wandb
import transformers

import nltk
nltk.download('punkt')

import torch
from torch.utils.data import DataLoader
import torch.nn as nn


from evaluation import evaluate, build_foreign_key_map_from_json



# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def postprocess_text(preds, labels):
    """Use this function to postprocess generations and labels before BLEU computation."""
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def evaluate_model(model, dataloader, tokenizer, max_seq_length, device):
    model.eval()

    all_preds = []
    all_labels = []

    avg_batch_acc = 0
    pred_file = open("pred.txt", "w")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            attention_mask = batch["attention_mask"].to(device)
            #token_type_ids = batch["token_type_ids"].to(device)
            
            generated_tokens = model.generate(
                input_ids,
                max_length=max_seq_length,
                num_beams=4
            )


            #logits = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

            #preds = torch.argmax(logits, dim=-1)
            #metric.add_batch(predictions=preds, references=labels)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # for row in decoded_preds:
            #   all_preds.append(row)

            labels = labels.tolist()


            for row in decoded_preds:
                all_preds.append("".join(row))
            

            new_labels = []
            for label_row in labels:
                new_labels.append([value for value in label_row if value != -100])
            
            decoded_labels = tokenizer.batch_decode(new_labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            
            bleu.add_batch(predictions=decoded_preds, references=decoded_labels)

    
    pred_file.write("\n".join(all_preds))

    pred_file.close()

    without_vals_scores = evaluate('gold.txt', 'pred.txt', 'database2', 'all', build_foreign_key_map_from_json('tables.json'), False, False, False)
    match_scores = evaluate('gold.txt', 'pred.txt', 'database', 'all', build_foreign_key_map_from_json('tables.json'), True, False, False)
    bleu_metric = bleu.compute()

    evaluation_results = {
        "eval/bleu": bleu_metric["score"],
        "eval/exec_without_val": without_vals_scores,
        "eval/match_scores": match_scores,
    }



    model.train()
    return evaluation_results, input_ids, decoded_preds, decoded_labels




def preprocess_function(examples, tokenizer, max_seq_length):
    

    inputs = examples['question']
    targets = examples['query']
    
    model_inputs = tokenizer(inputs, max_length=max_seq_length, padding="max_length", truncation=True)
    decoder_inputs = tokenizer(targets, max_length=max_seq_length, padding="max_length", truncation=True)
    target_ids = decoder_inputs.input_ids
    
    #decoder_input_ids = []

    # for target in target_ids:
    #     decoder_input_ids.append([tokenizer.bos_token_id] + target)
    #     labels.append(target + [tokenizer.eos_token_id])

    # model_inputs["decoder_input_ids"] = decoder_input_ids

    labels_with_ignore_index = []
    
    for labels_example in target_ids:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)
    
    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs



torch.cuda.empty_cache()
bleu = datasets.load_metric("sacrebleu")

dataset = load_dataset('spider')


max_seq_length=128
overwrite_cache=True
preprocessing_num_workers = 1
batch_size=len(dataset['train'])
num_train_epochs=30
device='cpu'
learning_rate=1e-4
weight_decay=0.01
lr_scheduler_type = 'polynomial'
num_warmup_steps = 200
max_train_steps = 20000
logging_steps=25
eval_every_step=100
output_dir = 'output_dir'

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
model = T5ForConditionalGeneration.from_pretrained('./output_dir')

column_names = dataset["train"].column_names

preprocess_function_wrapped = partial(
    preprocess_function,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
)


processed_datasets = dataset['train'].map(
    preprocess_function_wrapped,
    batched=True,
    num_proc=preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on dataset",
)

processed_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

train_dataset = processed_datasets
#eval_dataset = processed_datasets["validation"] if "validation" in processed_datasets else processed_datasets["test"]

# Log a few random samples from the training set:
for index in random.sample(range(len(train_dataset)), 2):
    print(f"Sample {index} of the training set: {train_dataset[index]}.")
    print(f"Decoded input_ids: {tokenizer.decode(train_dataset[index]['input_ids'])}")
    print(f"Decoded labels: {tokenizer.decode([label for label in train_dataset[index]['labels'] if label != -100])}")
    print("\n")



train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size
)

# eval_dataloader = DataLoader(
#     eval_dataset, shuffle=False, batch_size=batch_size
# )

model.to(device)


gold_file = open("gold.txt", "w")

gold_queries = []

for row in dataset['train']:
    gold_queries.append(row['query'] + '\t' + row['db_id'])
    
gold_file.write("\n".join(gold_queries))
  
gold_file.close()


run = wandb.init(project=f"CODET5_SQLNL")

global_step = 0

progress_bar = tqdm(range(len(train_dataloader) * num_train_epochs))

eval_results, last_input_ids, last_decoded_preds, last_decoded_labels = evaluate_model(
    model=model, 
    dataloader=train_dataloader, 
    tokenizer=tokenizer,            
    device=device,
    max_seq_length=max_seq_length,
)

wandb.log(
    {
    "eval/bleu": eval_results["eval/bleu"],
    "eval/match_scores": eval_results['eval/match_scores'], 
    "eval/exec_without_val": eval_results['eval/exec_without_val'], 
    #"eval/exact_match(vals)": eval_results["eval/exact_match(vals)"]
    },
    step=global_step,
)

run.finish()  # stop wandb run