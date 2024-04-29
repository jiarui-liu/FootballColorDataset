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

domains_dict = {
    'Lifestyle': 310,
    'Economy': 90,
    'Media & Technology': 242,
    'Social Dynamics': 183
}
domains = list(domains_dict.keys())

word_mapping = get_word_list("data/list_uk_us.csv", mode='dict')

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

for domain in domains:
    records_f = open(f"data/answers/{model}/records_answers_{domain}.txt", 'a')
    df = pd.read_csv(f"data/stylish_question_by_domain/{domain}.csv")
    # select top xx samples
    sampled_lst = df.to_dict(orient='records')
    sampled_lst = sampled_lst[:domains_dict[domain]]
    
    for idx, item in tqdm(enumerate(sampled_lst)):
        uk_str = remove_uk_markers(item['formatted_response'])
        
        # uk prompt
        prompt = [
            AnswerGen.prompt.format(question=uk_str)
        ]
        uk_res = call(
            prompt,
            llm_config_func,
            model_version=model,
            api_key=api_key,
            # org_id=1,
            org_id=org_id,
            model_path=model_path,
            verbose=True
        )
        
        # us prompt
        prompt = [
            AnswerGen.prompt.format(question=replace_uk_words_with_us(uk_str, word_mapping))
        ]
        us_res = call(
            prompt,
            llm_config_func,
            model_version=model,
            api_key=api_key,
            # org_id=0,
            org_id=org_id,
            model_path=model_path,
            verbose=True
        )
        
        info = {
            "index": idx,
            "domain": item['domain'],
            "topic": item['topic'],
            "step": "generate_answers",
            "question": item['formatted_response'],
            "uk_response": uk_res,
            "us_response": us_res
        }
        write_line_to_file(json.dumps(info), records_f)