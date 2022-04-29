from transformers import RobertaTokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from functools import partial

import random
import logging
import tqdm
from tqdm.auto import tqdm


import wandb
import transformers

import nltk
nltk.download('punkt')

import torch
from torch.utils.data import DataLoader
import torch.nn as nn


from evaluation import evaluate, build_foreign_key_map_from_json
from evaluation_orig import evaluate as evaluate2
from evaluation_orig import build_foreign_key_map_from_json as build_foreign_key_map_from_json2


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
            
            generated_tokens = model.model.generate(
                input_ids,
                max_length=max_seq_length,
                # beam_size=beam_size,
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

    
    pred_file.write("\n".join(all_preds))

    pred_file.close()
    without_vals_scores_orig = evaluate2('gold.txt', 'pred.txt', "/content/drive/MyDrive/database", 'match', build_foreign_key_map_from_json2('tables.json'))

    without_vals_scores = evaluate('gold.txt', 'pred.txt', 'database', 'match', build_foreign_key_map_from_json('tables.json'), False, False, False)
    with_vals_scores = evaluate('gold.txt', 'pred.txt', 'database', 'match', build_foreign_key_map_from_json('tables.json'), True, False, False)
    
    evaluation_results = {
        "orig_eval/exact_match": without_vals_scores_orig['all']['exact'],
        "eval/exact_match": without_vals_scores['all']['exact'],
        "eval/exact_match(vals)": with_vals_scores['all']['exact']
    }



    model.train()
    return evaluation_results, input_ids, decoded_preds, decoded_labels


class CodeT5_NLSQL(nn.Module):

  def __init__(self, model):
    super().__init__()

    self.model = model
    #self.input_size = model.config.to_dict()['hidden_size']
    #self.num_classes = num_classes

    #self.input_layer = nn.Linear(self.input_size)

  def forward(self, input_ids, attention_mask, labels=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs


tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small', force_download=True)

dataset = load_dataset('spider')

def preprocess_function(examples, tokenizer, max_seq_length):
    

    inputs = examples['question']
    targets = examples['query']
    
    model_inputs = tokenizer(inputs, max_length=max_seq_length, padding="max_length", truncation=True)
    target_ids = tokenizer(targets, max_length=max_seq_length, padding="max_length", truncation=True)
    target_ids = target_ids.input_ids
    
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

max_seq_length=128
overwrite_cache=True
preprocessing_num_workers = 8
batch_size=32
num_train_epochs=5
device='cuda'
learning_rate=1e-1
weight_decay=0.01
lr_scheduler_type = 'linear'
num_warmup_steps = 5000
max_train_steps = 20000
logging_steps=25
eval_every_step=1

column_names = dataset["train"].column_names

preprocess_function_wrapped = partial(
    preprocess_function,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
)


processed_datasets = dataset.map(
    preprocess_function_wrapped,
    batched=True,
    num_proc=preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on dataset",
)

processed_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"] if "validation" in processed_datasets else processed_datasets["test"]

# Log a few random samples from the training set:
for index in random.sample(range(len(train_dataset)), 2):
    print(f"Sample {index} of the training set: {train_dataset[index]}.")
    print(f"Decoded input_ids: {tokenizer.decode(train_dataset[index]['input_ids'])}")
    print(f"Decoded labels: {tokenizer.decode([label for label in train_dataset[index]['labels'] if label != -100])}")
    print("\n")

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size
)

eval_dataloader = DataLoader(
    eval_dataset, shuffle=False, batch_size=batch_size
)

nlsql_model = CodeT5_NLSQL(model)
nlsql_model.to(device)


gold_file = open("gold.txt", "w")

gold_queries = []

for row in dataset['validation']:
    gold_queries.append(row['query'] + '\t' + row['db_id'])
    
gold_file.write("\n".join(gold_queries))
  
gold_file.close()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)


lr_scheduler = transformers.get_scheduler(
    name=lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=max_train_steps,
)

run = wandb.init(project=f"CODET5_SQLNL")

global_step = 0

progress_bar = tqdm(range(len(train_dataloader) * num_train_epochs))


# iterate over epochs
for epoch in range(num_train_epochs):
    nlsql_model.train()  # make sure that model is in training mode, e.g. dropout is enabled

    # iterate over batches
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        # decoder_input_ids = batch["decoder_input_ids"].to(device)
        # key_padding_mask = batch["encoder_padding_mask"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = nlsql_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        global_step += 1

        wandb.log(
            {
                "train_loss": loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            },
            step=global_step,
        )


        if global_step % eval_every_step == 0:
            eval_results, last_input_ids, last_decoded_preds, last_decoded_labels = evaluate_model(
                model=nlsql_model,
                dataloader=eval_dataloader,
                tokenizer=tokenizer,
                device=device,
                max_seq_length=max_seq_length,

            )    
            wandb.log(
              {
               "orig_eval/exact_match": eval_results['orig_eval/exact_match'],     
               "eval/exact_match": eval_results['eval/exact_match'], 
               "eval/exact_match(vals)": eval_results["eval/exact_match(vals)"]
              },
              step=global_step,
            )
            print("Generation example:")
            random_index = random.randint(0, len(last_input_ids) - 1)
            print(f"Input sentence: {tokenizer.decode(last_input_ids[random_index], skip_special_tokens=True)}")
            print(f"Generated sentence: {last_decoded_preds[random_index]}")
            print(f"Reference sentence: {last_decoded_labels[random_index]}")


        if global_step % logging_steps == 0:
            # An extra training metric that might be useful for understanding
            # how well the model is doing on the training set.
            # Please pay attention to it during training.
            # If the metric is significantly below 80%, there is a chance of a bug somewhere.
            predictions = logits.argmax(-1)

            label_nonpad_mask = labels != tokenizer.pad_token_id
            num_words_in_batch = label_nonpad_mask.sum().item()

            accuracy = (predictions == labels).masked_select(label_nonpad_mask).sum().item() / num_words_in_batch

            wandb.log(
                {"train_batch_word_accuracy": accuracy},
                step=global_step,
            )
            
            # print("Generation example:")
            # random_index = random.randint(0, len(last_input_ids) - 1)
            # print(f"Input sentence: {tokenizer.decode(last_input_ids[random_index], skip_special_tokens=True)}")
            # print(f"Generated sentence: {last_decoded_preds[random_index]}")
            # print(f"Reference sentence: {last_decoded_labels[random_index][0]}")

        # if global_step % args.eval_every_steps == 0 or global_step == args.max_train_steps:
        #     eval_results, last_input_ids, last_decoded_preds, last_decoded_labels = evaluate_model(
        #         model=model,
        #         dataloader=eval_dataloader,
        #         target_tokenizer=target_tokenizer,
        #         device=args.device,
        #         max_seq_length=args.max_seq_length,
        #         generation_type=args.generation_type,
        #         beam_size=args.beam_size,
        #         max_eval_steps=args.max_eval_steps
        #     )
        #     # YOUR CODE ENDS HERE
        #     wandb.log(
        #         {
        #             "eval/bleu": eval_results["bleu"],
        #             "eval/generation_length": eval_results["generation_length"],
        #         },
        #         step=global_step,
        #     )
        #     logger.info("Generation example:")
        #     random_index = random.randint(0, len(last_input_ids) - 1)
        #     logger.info(f"Input sentence: {source_tokenizer.decode(last_input_ids[random_index], skip_special_tokens=True)}")
        #     logger.info(f"Generated sentence: {last_decoded_preds[random_index]}")
        #     logger.info(f"Reference sentence: {last_decoded_labels[random_index][0]}")


        #     logger.info("Saving model checkpoint to %s", args.output_dir)
        #     model.save_pretrained(args.output_dir)

        # if global_step >= args.max_train_steps:
        #     break

run.finish()  # stop wandb run