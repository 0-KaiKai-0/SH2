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

from dola import DoLa

def load_csv(file_path):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    '''
    Data format:

    ,full_prefix,doc_id,completion,contradiction_0,contradiction_1,contradiction_2,longest_completions,turncated_prefixes
    0,"As streaming television services continue to gain market share, there are a number of reasons why Netflix might be in trouble. Time Warner is taking its HBO content online, Amazon offers premium content for a monthly fee, and Hulu has reached nine million users. While these competitors may cause a bit of worry, it’s not the end of the world. Although Netflix has a huge amount of potential, the increased competition is unlikely to hurt its profitability.
    While the global pandemic last year caused a major shakeup in Hollywood, Netflix should not rest on its laurels. With a variety of rivals on the rise, it’s unlikely that it can continue to rely on its current performance. Despite the competition, the company has made a number of impactful moves across the board, including clamping down on password sharing. And in the coming years, Netflix should continue to grow and compete with new competitors.
    With more competitors entering the streaming space, Netflix is likely to face a more difficult time keeping its current market share. Disney has been investing heavily in the service and Amazon is expected to do the same. Both companies expect to add 35-40 million subscribers per year through 2024. Despite the competition, Netflix still remains the top streaming service. Its lack of original content has hurt its numbers in the last few quarters. Its only big original hit in the US was Cobra Kai, which only got four seasons. ",0,Whether or not it gets a second season of The Witcher is another question.,Whether or not it gets a second season of Stranger Things is another question.,Whether or not it gets a fifth season of The Witcher is another question.,Whether or not it gets a second season of Black Mirror is another question.,15.0,"As streaming television services continue to gain market share, there are a number of reasons why Netflix might be in trouble. Time Warner is taking its HBO content online, Amazon offers premium content for a monthly fee, and Hulu has reached nine million users. While these competitors may cause a bit of worry, it’s not the end of the world. Although Netflix has a huge amount of potential, the increased competition is unlikely to hurt its profitability.
    While the global pandemic last year caused a major shakeup in Hollywood, Netflix should not rest on its laurels. With a variety of rivals on the rise, it’s unlikely that it can continue to rely on its current performance. Despite the competition, the company has made a number of impactful moves across the board, including clamping down on password sharing. And in the coming years, Netflix should continue to grow and compete with new competitors.
    With more competitors entering the streaming space, Netflix is likely to face a more difficult time keeping its current market share. Disney has been investing heavily in the service and Amazon is expected to do the same. Both companies expect to add 35-40 million subscribers per year through 2024. Despite the competition, Netflix still remains the top streaming service. Its lack of original content has hurt its numbers in the last few quarters. Its only big original hit in the US was Cobra Kai, which only got four seasons. "

    '''
    list_data_dict = []
    df = pd.read_csv(file_path)
    # if 'news' in file_path:
    #     prefix_type = 'full_prefix'
    # else:
    prefix_type = 'turncated_prefixes'
    for idx in range(len(df)):
        item = dict(
            prefix=df[prefix_type][idx],
            completion=df['completion'][idx],
            contradiction_0=df['contradiction_0'][idx],
            contradiction_1=df['contradiction_1'][idx],
            contradiction_2=df['contradiction_2'][idx],
        )
        list_data_dict.append(item)
    return list_data_dict


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
parser.add_argument("--num-gpus", type=str, default="1")
parser.add_argument("--max_gpu_memory", type=int, default=27)
parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
parser.add_argument("--data-path", type=str, default="./strqa")
parser.add_argument("--output-path", type=str, default="./strqa_result")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--key-num", type=int, default=3)

args = parser.parse_args()
model_name = args.model_name
num_gpus = args.num_gpus
device = args.device


fp = args.data_path
list_data_dict = load_csv(fp)

llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
llm.set_stop_words(["Q:", "\end{code}"])


key_words = []
cnt = 0
with torch.no_grad():
    for sample in tqdm(list_data_dict):
        key_words.append(llm.key_words(sample['prefix'], key_num=args.key_num))
        cnt += 1
        if cnt <= 10:
            print(cnt - 1)
            print("prefix:")
            print(sample['prefix'])
            print("key word:")
            print(key_words[-1])

output_file = args.output_path
with open(output_file, 'w') as f:
    json.dump(key_words, f)

# # save results to a json file
# model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
# output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".json")
# with open(output_file, 'w') as f:
#     json.dump(result_dict, f)