def get_json_list(path):
    import json
    f = open(path, 'r')
    info = []
    for line in f.readlines():
        info.append(json.loads(line))
    return info

def format_other_domains_str(other_domains_lst):
    return ", ".join(other_domains_lst[:-1]) + ", and " + other_domains_lst[-1]

def write_line_to_file(in_str, f):
    f.write(in_str)
    f.write("\n")
    f.flush()

def load_questions(path):
    data = get_json_list(path)
    new_data = []
    for item in data:
        if item['step'] != 'process_questions':
            continue
        
        for response in item['response']:
            info = {
                "domain": item['domain'],
                "topic": item['topic'],
                "question": response
            }
            new_data.append(info)
    return new_data

def get_word_list(path, mode='list'):
    import pandas as pd
    df = pd.read_csv(path, sep=';')
    if mode == 'list':
        word_list_uk = df['uk_usage'].values.tolist()
        return word_list_uk
    elif mode == 'dict':
        word_mapping = df.set_index("uk_usage")['us_usage'].to_dict()
        return word_mapping

def check_words(in_str, word_list_uk):
    import nltk
    in_str = in_str.lower()
    uk_words = []
    for word in word_list_uk:
        if word in in_str:
            uk_words.append(word)
    return list(set(uk_words))

def filter_generated_questions(q_lst):
    # step 1: remove questions whose 'uk_words' == []
    from copy import deepcopy
    data_dict = {
        "Lifestyle": [],
        "Economy": [],
        "Media & Technology": [],
        "Social Dynamics": []
    }
    
    for item in q_lst:
        if item['step'] != 'process_questions':
            continue
        if item['uk_words'] == []:
            continue
        
        item_cp = deepcopy(item)
        data_dict[item['domain']].append(item_cp)
        
    # step 2: reversely sort the list based on len(item['uk_words'])
    for key in data_dict.keys():
        data_dict[key] = sorted(data_dict[key], key=lambda x: len(x['uk_words']), reverse=True)
    return data_dict

def count_uk_words(data_dict, domains_dict):
    from collections import Counter
    import json
    for key, val in data_dict.items():
        word_list = []
        for item in val:
            word_list.extend(item['uk_words'])
        res = dict(Counter(word_list))
        # print(res)
        res = dict(sorted(res.items(), key=lambda i: i[1], reverse=True))
        # print(res)
        print(key, json.dumps(res, indent=2))

def replace_uk_words_with_us(uk_str, word_mapping):
    import re
    # sort the dict keys by length in descending order
    sorted_uk_words = sorted(word_mapping.keys(), key=len, reverse=True)
    
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_uk_words)) + r')\b')
    
    # Function to replace each match with its US equivalent
    def replace_match(match):
        return word_mapping[match.group(0)]
    
    # Replace occurrences in the text using the pattern
    us_str = pattern.sub(replace_match, uk_str)
    
    return us_str

def remove_uk_markers(uk_str):
    uk_str = uk_str.replace("In the UK", "")
    uk_str = uk_str.replace(" in the UK", "")
    uk_str = uk_str.replace("In the uk", "")
    uk_str = uk_str.replace(" in the uk", "")
    uk_str = uk_str.replace("In the United Kingdom", "")
    uk_str = uk_str.replace(" in the United Kingdom", "")
    uk_str = uk_str.replace(" British", "")
    uk_str = uk_str.replace("British ", "")
    uk_str = uk_str.replace(" in Britain", "")
    uk_str = uk_str.replace(" the UK", "")
    uk_str = uk_str.replace(" the uk", "")
    
    return uk_str

def get_score_list(obj_path, subj_path):
    import pandas as pd
    import numpy as np
    import math
    # subjective questions
    df_subj = pd.read_csv(subj_path, sep=';')
    df_subj = df_subj.loc[df_subj['source'].isin(['GAS', 'WVS'])]
    print(df_subj.shape)
    subj_scores_diff = np.abs(df_subj['uk_score'].values - df_subj['us_score'].values).tolist()
    
    # objective questions
    data_obj = get_json_list(obj_path)
    df_obj = pd.DataFrame().from_records(data_obj)
    obj_scores_diff = ((df_obj['score'].values - 1.0) / (5.0 - 1.0)).tolist()
    
    # res = subj_scores_diff + obj_scores_diff
    # res = subj_scores_diff
    res = obj_scores_diff
    # remove nan
    res = [x for x in res if not math.isnan(x)]    
    return res

def get_permutation_score_list(obj_path, subj_path, mode='subj'):
    """
    mode: choose from 'subj', 'obj', 'both'
    """
    import pandas as pd
    import numpy as np
    import math
    # subjective questions
    df_subj = pd.read_csv(subj_path, sep=';')
    df_subj = df_subj.loc[df_subj['source'].isin(['GAS', 'WVS'])]
    print(df_subj.shape)
    subj_scores_diff = (1 - abs(df_subj['uk_score'].values - df_subj['us_score'].values)).tolist()
    
    # objective questions
    data_obj = get_json_list(obj_path)
    df_obj = pd.DataFrame().from_records(data_obj)
    obj_scores_diff = ((df_obj['score'].values - 1.0) / (5.0 - 1.0)).tolist()
    
    if mode == 'subj':
        res = subj_scores_diff
    elif mode == 'obj':
        res = obj_scores_diff
    elif mode == 'both':
        res = subj_scores_diff + obj_scores_diff
    # remove nan
    res = [x for x in res if not math.isnan(x)]    
    return res

def get_permutation_p(diff_lst, n_permutations=1000):
    import numpy as np
    diff_lst = np.array(diff_lst)
    # Calculate the observed test statistic
    observed_stat = np.mean(diff_lst)
    
    perm_stats = []

    for _ in range(n_permutations):
        # Randomly flip the signs of the differences
        signed_diffs = diff_lst * np.random.choice([-1, 1], size=diff_lst.size)
        # Calculate the mean of these permuted differences
        perm_stats.append(np.mean(signed_diffs))

    # Calculate the p-value
    perm_stats = np.array(perm_stats)
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
    
    print(f'Observed Statistic: {observed_stat}')
    print(f'p-value: {p_value}')
    return observed_stat, p_value

def get_similarity_p(sim_lst, threshold=0.8):
    import numpy as np
    from scipy.stats import binomtest
    sim_lst = np.array(sim_lst)
    n_above = np.sum(sim_lst > threshold)
    n_total = len(sim_lst)
    res = binomtest(n_above, n_total, p=0.5, alternative='greater')
    print("Number of samples above 0.8:", n_above)
    print("Total number of samples:", n_total)
    print("P-statistic:", res.statistic)
    print("P-value:", res.pvalue)

    # Interpretation of results
    alpha = 0.05
    if res.pvalue < alpha:
        print("Reject the null hypothesis: There is a statistically significant number of observations above 0.8")
    else:
        print("Fail to reject the null hypothesis: There is not a statistically significant number of observations above 0.8")
    return res.statistic, res.pvalue

def get_stats_list(obj_path, subj_path):
    import pandas as pd
    import numpy as np
    import math
    import nltk
    # subjective questions
    words_set = set()
    df_subj = pd.read_csv(subj_path, sep=';')
    df_subj = df_subj.loc[df_subj['source'].isin(['GAS', 'WVS'])]
    print(df_subj.shape)
    subj_scores_diff = [len(nltk.word_tokenize(x)) for x in df_subj['question'].values.tolist()]
    [words_set.add(x) for y in df_subj['question'].values.tolist() for x in nltk.word_tokenize(y)]
    
    # objective questions
    data_obj = get_json_list(obj_path)
    df_obj = pd.DataFrame().from_records(data_obj)
    obj_x_lst = df_obj['question'].values.tolist()
    
    tmp_dict = {
        "Economy": pd.read_csv("data/stylish_question_by_domain/Economy.csv"),
        "Lifestyle": pd.read_csv("data/stylish_question_by_domain/Lifestyle.csv"),
        "Media & Technology": pd.read_csv("data/stylish_question_by_domain/Media & Technology.csv"),
        "Politics": pd.read_csv("data/stylish_question_by_domain/Politics.csv"),
        "Social Dynamics": pd.read_csv("data/stylish_question_by_domain/Social Dynamics.csv")
    }
    obj_q_lst = []
    for idx, row in df_obj.iterrows():
        index = row['index']
        obj_q_lst.append(tmp_dict[row['domain']].iloc[index]['question'])
        
    obj_scores_diff = [len(nltk.word_tokenize(x)) for x in obj_q_lst if type(x) == str]

    for y in obj_q_lst:
        if type(y) == str:
            for x in nltk.word_tokenize(y):
                words_set.add(x)
    
    res = subj_scores_diff + obj_scores_diff
    # remove nan
    res = [x for x in res if not math.isnan(x)]    
    print("Length of set", len(words_set))
    return res