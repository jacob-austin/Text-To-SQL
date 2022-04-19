import pickle
import argparse
import os
import random

from sql_util.dbinfo import get_all_db_info_path

from fuzz.neighbor import generate_neighbor_queries_path
from fuzz.fuzz import generate_random_db_with_queries_wrapper

from sql_util.run import exec_db_path
from sql_util.eq import result_eq

import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train machine translation transformer model")

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help=("Where to store the final model. "
              "Should contain the source and target tokenizers in the following format: "
              r"output_dir/{source_lang}_tokenizer and output_dir/{target_lang}_tokenizer. "
              "Both of these should be directories containing tokenizer.json files."
        ),
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=("Where to store the final model. "
              "Should contain the source and target tokenizers in the following format: "
              r"output_dir/{source_lang}_tokenizer and output_dir/{target_lang}_tokenizer. "
              "Both of these should be directories containing tokenizer.json files."
        ),
    )

    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=2,
        required=False,
        help=("Where to store the final model. "
                "Should contain the source and target tokenizers in the following format: "
                r"output_dir/{source_lang}_tokenizer and output_dir/{target_lang}_tokenizer. "
                "Both of these should be directories containing tokenizer.json files."
        ),
    )

    return parser.parse_args()

def main():
    args = parse_args()

    dataset_file_path = 'dataset/classical_test.pkl'
    output_lines = []


    with open(args.input_path, "rb") as file:
        dataset = pickle.load(file)

    for t, ex in enumerate(dataset[:100]):
        print(t)
        original_database_path = ex['db_path']
        
        neighbors = generate_neighbor_queries_path(original_database_path, ex['query'])
        random.shuffle(neighbors)

        undistinguished_neighbors = set(neighbors)
        
        num_added = 0
        output_lines.append(ex['query'] + '\t' + ex['db_id'])

        for u in set(undistinguished_neighbors):
            sampled_database_w_path = 'database/db%d.sqlite' % t
            generate_random_db_with_queries_wrapper((original_database_path, sampled_database_w_path, [ex['query']], {}))

            gold_flag, gold_denotation = exec_db_path(sampled_database_w_path, ex['query'])
            u_flag, u_denotation = exec_db_path(sampled_database_w_path, u)
            
            if(gold_flag == 'exception' or u_flag == 'exception'):
                continue
            if not result_eq(gold_denotation, u_denotation, order_matters=False) and len(u_denotation) > 0:
                if num_added >= args.num_neighbors: break

                undistinguished_neighbors.remove(u)
                output_lines.append(u + "\t" + ex['db_id'])
                num_added += 1
        
        output_lines.append("")

    with open(os.path.join(args.output_path), 'w+') as file:
        for line in output_lines:
            file.write(line + "\n")
    
if __name__ == "__main__":
    main()