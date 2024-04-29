import re

class AAEPromptGen:
    prompt = """Below is an essay prompt written in Standard American English. Please rewrite it in African-American English, ensuring that the meaning of the prompt and the question type remain unchanged and that it still poses a question in the same way.

Standard American English: {sae_prompt}

African-American English:
"""


class ESLPromptGen:
    prompt = """Below is an essay prompt written in Standard American English. Please introduce more than two common errors that are typical of English as a Second Language (ESL) speakers into the prompt, ensuring that the meaning of the prompt and the question type remain unchanged and that it still poses a question in the same way. Common errors may arise from grammar, syntax, vocabulary, cultural nuances, and spelling.

Standard American English: {sae_prompt}

ESL English:
"""

class EssayGen:
    prompt = """Please write an essay of approximately 300-350 words in response to the following question:

Question: {question}

Begin your essay below:
"""