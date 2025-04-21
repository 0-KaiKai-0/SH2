import re
import os
import json
import transformers
import torch
from tqdm import tqdm, trange
import argparse
import pandas as pd

import ssl
import urllib.request
import zipfile

from modeling import SH2

def load_csv(file_path, is_gzip=False):
    # input file is in csv format, can be loaded by pandas
    # required columns: [Question] only

    open_func = open if not is_gzip else gzip.open
    list_data = []
    with open_func(file_path, 'r') as f:
        df = pd.read_csv(f)
        for idx in range(len(df)):
            data = {'question': df['Question'][idx], 
                    'answer_best': df['Best Answer'][idx],
                    'answer_true': df['Correct Answers'][idx],
                    'answer_false': df['Incorrect Answers'][idx]}
            list_data.append(data)

    return list_data



parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
parser.add_argument("--num-gpus", type=str, default="1")
parser.add_argument("--max_gpu_memory", type=int, default=27)
parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
parser.add_argument("--data-path", type=str, default="./tfqa")
parser.add_argument("--output-path", type=str, default="./tfqa_result")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--key-num", type=int, default=3)

args = parser.parse_args()
model_name = args.model_name
num_gpus = args.num_gpus
device = args.device


fp = os.path.join(args.data_path, 'TruthfulQA.csv')
if not os.path.exists(fp):
    download_url(
        'https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv', args.data_path)

list_data_dict = load_csv(fp)


llm = SH2(model_name, device, num_gpus, args.max_gpu_memory)
stop_word_list = ["Q:"]
llm.set_stop_words(stop_word_list)

question = []
question.append("What is human life expectancy in the United States?")
question.append("Who was president of the United States in 1955?")
question.append("Which party did he belong to?")
question.append("What is the square root of banana?")
question.append("How does a telescope work?")
question.append("Where were the 1992 Olympics held?")

key_words = []
with torch.no_grad():
    for q in question:
        print(q)
        key_words.append(llm.key_words(q, key_num=args.key_num))
        print(key_words[-1])

cnt = 0
with torch.no_grad():
    for sample in tqdm(list_data_dict):
        key_words.append(llm.key_words(sample['question'], key_num=args.key_num))
        cnt += 1

output_file = args.output_path
with open(output_file, 'w') as f:
    json.dump(key_words, f)

# # save results to a json file
# model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
# output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".json")
# with open(output_file, 'w') as f:
#     json.dump(result_dict, f)