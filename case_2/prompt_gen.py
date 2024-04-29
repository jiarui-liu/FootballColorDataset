import json
import pandas as pd
from tqdm import tqdm
from prompts import *
from llm import call
from helper import *

def llm_config_func(llm):
    llm.temperature = 0.0
    llm.max_tokens = 4096
    return llm

# AAE prompt generation
model = 'gpt-4-1106-preview'
records_f = open(f"prompts/aae_prompt.txt", 'a')
df = pd.read_csv(f"prompts/prompt.csv")
# select top xx samples
sampled_lst = df.to_dict(orient='records')

for idx, item in tqdm(enumerate(sampled_lst)):
    
    prompt = [
        AAEPromptGen.prompt.format(sae_prompt=item['sae_prompt'])
    ]
    res = call(
        prompt,
        llm_config_func,
        model_version='gpt-4-1106-preview',
        org_id=1,
        verbose=False
    )
    
    
    info = {
        "index": idx,
        "sae_prompt": item['sae_prompt'],
        "aae_prompt": res
    }
    write_line_to_file(json.dumps(info), records_f)

# ESL prompt generation
records_f = open(f"prompts/esl_prompt.txt", 'a')
df = pd.read_csv(f"prompts/prompt.csv")
# select top xx samples
sampled_lst = df.to_dict(orient='records')

for idx, item in tqdm(enumerate(sampled_lst)):
    
    prompt = [
        ESLPromptGen.prompt.format(sae_prompt=item['sae_prompt'])
    ]
    res = call(
        prompt,
        llm_config_func,
        model_version='gpt-4-1106-preview',
        org_id=0,
        verbose=False
    )
    
    
    info = {
        "index": idx,
        "sae_prompt": item['sae_prompt'],
        "esl_prompt": res
    }
    write_line_to_file(json.dumps(info), records_f)