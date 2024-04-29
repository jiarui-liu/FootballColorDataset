import json
import pandas as pd
from tqdm import tqdm
from prompts import *
from llm import call
from helper import *


q_lst = load_questions("data/records_questions.txt")
df = pd.DataFrame().from_records(q_lst)

domains_dict = {
    'Lifestyle': 400,
    'Economy': 300,
    'Media & Technology': 400,
    'Social Dynamics': 400
}
sampled_lst = []
for key, val in domains_dict.items():
    domain_df = df.loc[df['domain'] == key].sample(n=val, random_state=1)
    sampled_lst.append(domain_df)
sampled_df = pd.concat(sampled_lst)
# sampled_df

sampled_lst = sampled_df.to_dict(orient='records')
# sampled_lst[10]

def llm_config_func(llm):
    llm.temperature = 0
    llm.max_tokens = 4096
    return llm

records_f = open("data/records_stylish_questions.txt", 'a')
word_list_uk = get_word_list("data/list_uk_us.csv")
for item in tqdm(sampled_lst[5:]):
    prompt = [
        StylishQuestionGen.system_prompt,
        StylishQuestionGen.prompt.format(
            word_list_uk="\n".join(word_list_uk),
            question=item['question']
        )
    ]
    res = call(
        prompt,
        llm_config_func,
        model_version='gpt-4-1106-preview',
        org_id=0,
        verbose=False
    )
    
    info = {
        "domain": item['domain'],
        "topic": item['topic'],
        "step": "generate_questions",
        "question": item['question'],
        "response": res,
        "prompt": prompt
    }
    write_line_to_file(json.dumps(info), records_f)
    
    try:
        formatted_res, uk_words = StylishQuestionGen.process_response(res, word_list_uk)
        info = {
            "domain": item['domain'],
            "topic": item['topic'],
            "step": "process_questions",
            "question": item['question'],
            "response": res,
            "formatted_response": formatted_res,
            "uk_words": uk_words,
            "prompt": prompt
        }
        write_line_to_file(json.dumps(info), records_f)
    except:
        print(f"####process questions step failed")