# Ref: https://github.com/kojima-takeshi188/zero_shot_cot

import re
import os
import json
import random
import torch
import numpy as np
import transformers
from tqdm import tqdm, trange
import argparse
from collections import defaultdict, Counter
import glob
import sys

import ssl
import urllib.request
import zipfile
import tiktoken

from dola import DoLa

transformers.logging.set_verbosity(40)

DEBUG = False

def num_tokens_from_message(message, model="davinci"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(message))
    return num_tokens


def truncate_message(prompt1, prompt2, model="davinci"):
    if num_tokens_from_message(prompt1 + prompt2, model) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split()[:-1])
    prompt = prompt1 + prompt2
    return prompt


demo_keys = []
def load_jsonl(file_path, pondering=None, keys_path=None):
    global demo_keys
    if args.keys_path is not None:
        with open(args.keys_path, "r", encoding="utf-8") as f:
            key_words = json.load(f)

    list_data_dict = {}
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        data = data[:1000]

        candicates = ["hallucinated_summary", "right_summary"]
        ground_truths = ["Yes", "No"]
        for j in range(len(candicates)):
            list_data_dict[j] = []
            for idx in range(len(data)):
                response = "\n#Summary#: " + data[idx][candicates[j]] + "\n#Your Judgement#:"
                ground_truth = ground_truths[j]

                new_item = dict(
                    context="#Document#: " + data[idx]["document"],
                    response=response,
                    answer=ground_truth
                )
                if pondering == 'pause':
                    new_item['response'] = "\n#Pondering#: " + "." * args.pause_num + response
                elif pondering == 'repeat':
                    new_item['response'] = "\n#Pondering#: " + data[idx]["document"] + response
                elif pondering == 'hard':
                    if keys_path is not None:
                        demo_keys = key_words[:3]
                        new_item['response'] = "\n#Pondering#: " + key_words[3 + idx] + response
                elif pondering == 'hard-prepend':
                    if keys_path is not None:
                        demo_keys = key_words[:3]
                        new_item['context'] = "#Pondering#: " + key_words[3 + idx] + "\n" + new_item['context']
                list_data_dict[j].append(new_item)

    return list_data_dict


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


def create_demo_text(pondering=None):
    prompt, context, response, answer = [], [], [], []

    prompt.append("You are trying to determine if the summary is factual but some information cannot be directly inferred or entailed from the document.")
    context.append("#Document#: The panther chameleon was found on Monday by a dog walker in the wooded area at Marl Park. It had to be put down after X-rays showed all of its legs were broken and it had a deformed spine. RSPCA Cymru said it was an \"extremely sad example of an abandoned and neglected exotic pet\". Inspector Selina Chan said: \"It is a possibility that the owners took on this animal but were unable to provide the care he needs and decided to release him to the wild. \"We are urging potential owners of exotic animals to thoroughly research what is required in the care of the particular species before taking one on. \"Potential owners need to make sure they can give their animal the environment it needs and they have the facilities, time, financial means and long-term commitment to maintain a good standard of care, as required under the Animal Welfare Act 2006.\" She added it was illegal to release non-native species into the wild.")
    response.append("#Summary#: A chameleon that was found in a Cardiff park has been put down after being abandoned and neglected by its owners.")
    answer.append("#Your Judgement#: Yes")

    prompt.append("You are trying to determine if there exists some non-factual and incorrect information in the summary.  ")
    context.append("#Document#: The city was brought to a standstill on 15 December last year when a gunman held 18 hostages for 17 hours. Family members of victims Tori Johnson and Katrina Dawson were in attendance. Images of the floral tributes that filled the city centre in the wake of the siege were projected on to the cafe and surrounding buildings in an emotional twilight ceremony. Prime Minister Malcolm Turnbull gave an address saying a \"whole nation resolved to answer hatred with love\". \"Testament to the spirit of Australians is that with such unnecessary, thoughtless tragedy, an amazing birth of mateship, unity and love occurs. Proud to be Australian,\" he said. How the Sydney siege unfolded New South Wales Premier Mike Baird has also announced plans for a permanent memorial to be built into the pavement in Martin Place. Clear cubes containing flowers will be embedded into the concrete and will shine with specialised lighting. It is a project inspired by the massive floral tributes that were left in the days after the siege. \"Something remarkable happened here. As a city we were drawn to Martin Place. We came in shock and in sorrow but every step we took was with purpose,\" he said on Tuesday.")
    response.append("#Summary#: Crowds have gathered in Sydney's Martin Place to honour the victims of the Lindt cafe siege, one year on.")
    answer.append("#Your Judgement#: No")

    prompt.append("You are trying to determine if there is a factual contradiction between the summary and the document.")
    context.append("#Document#: Christopher Huxtable, 34, from Swansea, had been missing since the collapse in February. His body was found on Wednesday and workers who carried out the search formed a guard of honour as it was driven from the site in the early hours of the morning. Ken Cresswell, 57, and John Shaw, 61, both from Rotherham, remain missing. The body of a fourth man, Michael Collings, 53, from Brotton, Teesside, was previously recovered from the site. Swansea East MP Carolyn Harris, who has been involved with the family since the incident, said they still did not know all the facts about the collapse. She said: \"I feel very sad. My heart and my prayers go out to the family who have waited desperately for Christopher's body to be found. They can finally have closure, and say goodbye to him and grieve his loss. \"But let's not forget that there's two other families who are still waiting for their loved ones to be returned.\" The building was due for demolition when it partially collapsed in February.")
    response.append("#Summary#: The body of a man whose body was found at the site of the Swansea Bay Power Station collapse has been removed from the site.")
    answer.append("#Your Judgement#: Yes")

    # Concatenate demonstration examples ...
    demo_text = "I want you act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.\n\n"
    for i in range(len(context)):
        if pondering is None:
            demo_text += prompt[i] + "\n" + context[i] + "\n" + \
                         response[i] + "\n" + answer[i] + "\n\n"
        elif pondering == 'pause':
            demo_text += prompt[i] + "\n" + context[i] + "\n#Pondering#: " + "." * args.pause_num + "\n" + \
                         response[i] + "\n" + answer[i] + "\n\n"
        elif pondering == 'repeat':
            demo_text += prompt[i] + "\n" + context[i] + "\n#Pondering#: " + question[i] + "\n" + \
                         response[i] + "\n" + answer[i] + "\n\n"
        elif pondering == 'hard':
            demo_text += prompt[i] + "\n" + context[i] + "\n#Pondering#: " + demo_keys[i] + "\n" + \
                         response[i] + "\n" + answer[i] + "\n\n"
        elif pondering == 'hard-prepend':
            demo_text += prompt[i] + "\n#Pondering#: " + demo_keys[i] + "\n" + context[i] + "\n" + \
                         response[i] + "\n" + answer[i] + "\n\n"

    return demo_text


def build_prompt(context, response, pondering=None):
    demo = create_demo_text(pondering)
    prompt = demo + context
    input_text_prompt = truncate_message(prompt, response)
    return input_text_prompt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./strqa")
    parser.add_argument("--output-path", type=str, default="./strqa_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--keys-path", type=str, default=None)
    parser.add_argument("--pondering", type=str, default=None)
    parser.add_argument("--pause-num", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=10)
    
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # set seed
    set_seed(args.seed)

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to

    # Get test file
    fp = args.data_path
    if not os.path.exists(fp):
        raise ValueError(f"Test file {fp} does not exist.")

    list_data_dict = load_jsonl(fp)
    if args.pondering is not None:
        list_data_dict_keys = load_jsonl(fp, pondering=args.pondering, keys_path=args.keys_path)

    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
    stop_word_list = ["#Document#:", "#Pondering#:"]
    llm.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1
    elif len(early_exit_layers) == 2:
        print(f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
        mode = "dola-static"
        mature_layer = early_exit_layers[1]
        premature_layer = early_exit_layers[0]
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    else:
        print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mode = "dola"
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        premature_layer_dist = {l:0 for l in candidate_premature_layers}
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
  
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, 
                            do_sample=args.do_sample, 
                            top_p=args.top_p, 
                            top_k=args.top_k, 
                            temperature=args.temperature, 
                            repetition_penalty=args.repetition_penalty, 
                            mode=mode, 
                            mature_layer=mature_layer, 
                            premature_layer=premature_layer, 
                            candidate_premature_layers=candidate_premature_layers, 
                            relative_top=args.relative_top,
                            pondering=args.pondering,
                            alpha=args.alpha)
    
    output_path = args.output_path
    candicates = ["hallucinated_summary", "right_summary"]
    corrects ,incorrects = [], []
    for j in range(len(candicates)):
        print("="*20 + candicates[j] + "="*20)
        correct = 0
        incorrect = 0
        for idx in tqdm(range(len(list_data_dict[j]))):
            sample = list_data_dict[j][idx]
            if args.pondering is None:
                input_text_keys = None
            else:
                sample_keys = list_data_dict_keys[j][idx]
                input_text_keys = build_prompt(sample_keys['context'], sample_keys['response'], pondering=args.pondering)

            input_text = build_prompt(sample['context'], sample['response'])

            model_completion, c_dist = llm.generate(input_text, input_text_keys=input_text_keys, **generate_kwargs)
            
            for stop_word in stop_word_list:
                length_to_remove = len(stop_word)
                if model_completion[-length_to_remove:] == stop_word:
                    model_completion = model_completion[:-length_to_remove]
            model_completion = model_completion.strip()
            ans = model_completion.replace(".", "")

            if mode == "dola":
                for k, v in c_dist.items():
                    premature_layer_dist[k] += v
            
            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"document": sample['context'], "summary": sample['response'], "ground_truth": sample['answer'], "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(idx))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"document": sample['context'], "summary": sample['response'], "ground_truth": sample['answer'], "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"document": sample['context'], "summary": sample['response'], "ground_truth": sample['answer'], "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if sample['answer'] == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(idx))
            dump_jsonl(gen, output_path, append=True)

        print('{}: {} correct samples, {} incorrect samples, Accuracy: {}'.format(candicates[j], correct, incorrect, correct / len(list_data_dict[j])))
        corrects.append(correct)
        incorrects.append(incorrect)
        
    print("=" * 50)
    correct, incorrect, total = 0, 0, 0
    for j in range(len(candicates)):
        print('{}: {} correct samples, {} incorrect samples, Accuracy: {}'.format(candicates[j], corrects[j], incorrects[j], corrects[j] / len(list_data_dict[j])))
        correct += corrects[j]
        incorrect += incorrects[j]
        total += len(list_data_dict[j])
    print('Total: {} correct samples, {} incorrect samples, acc_H {}, acc_A: {}'.format(correct, incorrect, 2*corrects[0]*corrects[1]/(correct*len(list_data_dict[0])), correct / total))
    precision = corrects[0] / (corrects[0] + incorrects[1])
    recall = corrects[0] / len(list_data_dict[0])
    F1 = (2 * precision * recall) / (precision + recall)
    print('Precision: {}, recall: {}, F1: {}'.format(precision, recall, F1))