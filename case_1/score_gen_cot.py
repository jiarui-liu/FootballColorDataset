import json
import random
import pandas as pd
from tqdm import tqdm
from prompts import *
from llm import call
from helper import *
import sys

def llm_config_func(llm):
    llm.temperature = 0.0
    llm.max_tokens = 1024
    return llm

domains_dict = {
    'Lifestyle': 310,
    'Economy': 90,
    'Media & Technology': 242,
    'Social Dynamics': 183
}
domains = list(domains_dict.keys())

model = "gpt-4-1106-preview"
org_id = [0,1]
api_key = None
model_path=None

gen_models = [
    # "alpaca",
    "gpt-4-1106-preview",
    # "llama2_7b",
    # "llama2_13b",
    # "vicuna_7b",
    # "vicuna_13b",
    # "llama2_70b"
]

for gen_model in gen_models:
    records_cot_f = open(f"data/scores/records_answers_cot_upd_{gen_model}.txt", 'a')
    records_f = open(f"data/scores/records_answers_noncot_upd_{gen_model}.txt", 'a')
    for domain in domains:
        data = get_json_list(f"data/cot_answers_upd/{gen_model}/records_answers_{domain}.txt")
        
        for idx, item in tqdm(enumerate(data)):
            sentence_lst = [item['uk_response'], item['us_response']]
            random.shuffle(sentence_lst)
            
            prompt = [
                ScoreGen.prompt.format(sentence1=sentence_lst[0], sentence2=sentence_lst[1])
            ]
            res = call(
                prompt,
                llm_config_func,
                model_version=model,
                api_key=api_key,
                # org_id=1,
                org_id=org_id,
                model_path=model_path,
                verbose=True
            )
            
            score = ScoreGen.process_response(res)
            
            info = {
                "index": item['index'],
                "domain": item['domain'],
                "topic": item['topic'],
                "step": "scores_answers",
                "question": item['question'],
                "uk_response": item['uk_response'],
                "us_response": item['us_response'],
                "score_response": res,
                "score": score
            }
            write_line_to_file(json.dumps(info), records_f)
            
            
            
            
            sentence_lst = [item['uk_response_cot'], item['us_response_cot']]
            random.shuffle(sentence_lst)
            
            prompt = [
                ScoreGen.prompt.format(sentence1=sentence_lst[0], sentence2=sentence_lst[1])
            ]
            res = call(
                prompt,
                llm_config_func,
                model_version=model,
                api_key=api_key,
                # org_id=1,
                org_id=org_id,
                model_path=model_path,
                verbose=True
            )
            
            score = ScoreGen.process_response(res)
            
            info = {
                "index": item['index'],
                "domain": item['domain'],
                "topic": item['topic'],
                "step": "scores_answers",
                "question": item['question'],
                "uk_response": item['uk_response_cot'],
                "us_response": item['us_response_cot'],
                "score_response": res,
                "score": score
            }
            write_line_to_file(json.dumps(info), records_cot_f)