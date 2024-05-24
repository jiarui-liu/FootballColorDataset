# [Implicit Personalization in Language Models: A Systematic Study](https://arxiv.org/abs/2405.14808)


## Dataset Construction Procedure


### Installation

```bash
pip install -r requirements.txt
```

`vllm/`: Set up the inference interface of LLMs via [VLLM](https://github.com/vllm-project/vllm).

### Case 1: Cultural Adaptivity

Our AmbrQA dataset includes 825 objective questions and 825 subjective questions.

Follow the steps below to reproduce the dataset construction for objective questions:
```python
# generate objective questions
python3 question_gen.py
# generate LLMs' answers to objective questions
python3 answer_gen.py
# score the LLM-generated answers
python3 score_gen.py
```

To reproduce the dataset construction for subjective questions, refer to [Nils' implementation for case 1](https://github.com/sirnyls/AutoPersonalizationBench/tree/main/case1_CulturalSensitivity).

Check `hypo_test.ipynb` for the hypothesis testing.

### Case 2: Education Disparity

Follow the steps below to reproduce the dataset construction:
```python
# generate essay prompts in three different writing styles
python3 prompt_gen.py
# generate LLMs' essay responses to the prompts
python3 essay_gen.py
```
To reproduce the scoring of generated essays, refer to [Nils' implementation for case 2](https://github.com/sirnyls/AutoPersonalizationBench/tree/main/case2_EducationalDisparity).

Check `hypo_test.ipynb` for the hypothesis testing.

### Case 3: Echo Chamber Test

To reproduce the dataset construction procedure, check [Nils' implementation for case 3](https://github.com/sirnyls/AutoPersonalizationBench/tree/main/case3_Echochamber).

Check `hypo_test.ipynb` for the hypothesis testing.

## Citation

If you find our work useful, please give us a star and cite as follows :)

```bibtex
@misc{jin2024implicit,
      title={Implicit Personalization in Language Models: A Systematic Study}, 
      author={Zhijing Jin and Nils Heil and Jiarui Liu and Shehzaad Dhuliawala and Yahang Qi and Bernhard Schölkopf and Rada Mihalcea and Mrinmaya Sachan},
      year={2024},
      eprint={2405.14808},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```