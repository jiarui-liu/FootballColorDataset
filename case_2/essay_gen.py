import json
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


model = sys.argv[1]

if 'gpt' in model:
    org_id = [0,1]
    api_key = None
    model_path=None
elif model == 'llama2_7b':
    api_key = "EMPTY"
    org_id = "http://127.0.0.1:2525/v1"
    model_path = "meta-llama/Llama-2-7b-chat-hf"
elif model == 'llama2_13b':
    api_key = "EMPTY"
    org_id = "http://127.0.0.1:2526/v1"
    model_path = "meta-llama/Llama-2-13b-chat-hf"
elif model == 'llama2_70b':
    api_key = "EMPTY"
    org_id = "http://127.0.0.1:9580"
    model_path = "meta-llama/Llama-2-70b-chat-hf"
elif model == 'vicuna_7b':
    api_key = "EMPTY"
    org_id = "http://127.0.0.1:9797/v1"
    model_path = "lmsys/vicuna-7b-v1.5"
elif model == 'vicuna_13b':
    api_key = "EMPTY"
    org_id = "http://127.0.0.1:9798/v1"
    model_path = "lmsys/vicuna-13b-v1.5"
elif model == 'alpaca':
    api_key = "EMPTY"
    org_id = "http://127.0.0.1:6767/v1"
    model_path = "chavinlo/alpaca-native"
print(model)

q_type_dict = {
    'sae': pd.read_csv("prompts/prompt.csv").to_dict(orient='records'),
    'aae': get_json_list("prompts/aae_prompt.txt"),
    'esl': get_json_list("prompts/esl_prompt.txt")
}

sampled_lst = []
for sae, aae, esl in zip(q_type_dict['sae'], q_type_dict['aae'], q_type_dict['esl']):
    info = {
        'index': aae['index'],
        'sae_prompt': sae['sae_prompt'],
        'aae_prompt': aae['aae_prompt'],
        'esl_prompt': esl['esl_prompt']
    }
    sampled_lst.append(info)

records_f = open(f"essays/records_{model}_cont.txt", 'a')

for idx, item in tqdm(enumerate(sampled_lst[290:])):
    
    # sae prompt
    prompt = [
        EssayGen.prompt.format(question=item['sae_prompt'])
    ]
    sae_res = call(
        prompt,
        llm_config_func,
        model_version=model,
        api_key=api_key,
        # org_id=1,
        org_id=org_id,
        model_path=model_path,
        verbose=True
    )
    
    # aae prompt
    prompt = [
        EssayGen.prompt.format(question=item['aae_prompt'])
    ]
    aae_res = call(
        prompt,
        llm_config_func,
        model_version=model,
        api_key=api_key,
        # org_id=1,
        org_id=org_id,
        model_path=model_path,
        verbose=True
    )
    
    # esl prompt
    prompt = [
        EssayGen.prompt.format(question=item['esl_prompt'])
    ]
    esl_res = call(
        prompt,
        llm_config_func,
        model_version=model,
        api_key=api_key,
        # org_id=1,
        org_id=org_id,
        model_path=model_path,
        verbose=True
    )
    
    info = {
        "index": item['index'],
        "sae_prompt": item['sae_prompt'],
        "aae_prompt": item['aae_prompt'],
        "esl_prompt": item["esl_prompt"],
        "sae_essay": sae_res,
        "aae_essay": aae_res,
        "esl_essay": esl_res
    }
    write_line_to_file(json.dumps(info), records_f)