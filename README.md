# Dataset creation procedure of the Football Colo(u)r paper

`vllm/`: Set up the inference interface of LLMs via [VLLM](https://github.com/vllm-project/vllm).

## Case 1: Cultural Adaptivity

Our AmbrQA dataset includes 825 objective questions and 825 subjective questions. Scores of subjective questions are from Nils' implementation.

Steps:
1. `question_gen.py`
2. `answer_gen.py`
3. `score_gen.py`


## Case 2

Steps:
1. `prompt_gen.py`
2. `essay_gen.py`

Nils obtain scores of the generated essays based on his code.