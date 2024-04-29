import re

domains = [
    "Lifestyle",
    "Economy",
    "Media & Technology",
    "Politics",
    "Social Dynamics"
]

class QuestionGen:
    system_prompt = """You are an expert in identifying subtle differences between American and British cultures across various domains."""
    
    # topics based on domains
    topic_prompt = """Please generate {num_topics} topics related to {domain} that do not overlap with {other_domains}. Each topic should appeal to both American and British audiences. Ensure each topic is concise, limited to four words. Please provide each topic on a separate line.
"""

    def process_topics(response):
        response = response.strip()
        response_lst = [re.sub(r'^\d+\.', '', item).strip().lower() for item in response.split("\n")]
        return response_lst
    
    topic_result = {
        'Lifestyle': ["festive holiday traditions compared", "schooling systems: a study", "slang and language evolution", "national cuisine and recipes", "home decorating styles contrasted", "public transportation experiences", "healthcare systems: understanding differences", "outdoor recreation and sports", "eating habits and etiquette", "climate influence on lifestyle", "gardening practices and customs", "fashion trends through ages", "fitness and wellness approaches", "pet ownership cultures", "local attractions and tourism", "literature and reading preferences", "diy culture and hobbies", "music scene and genres", "religious observances and holidays", "youth culture and pastimes"],
        'Economy': ["transatlantic trade agreements impact", "comparative currency valuation trends", "tech startup investment growth", "real estate market divergence", "financial services regulation changes", "stock market performance analysis", "unemployment rate historical comparisons", "inflation patterns and predictions", "international business tax strategies", "renewable energy economic influence", "consumer spending habits shifts", "public debt management approaches", "economic impact of tourism", "retail sector evolution study", "gig economy workforce expansion", "cross-border investment challenges", "pension reform and security", "central bank policy effects", "agricultural subsidies and trade", "higher education funding models"],
        'Media & Technology': ["streaming services content comparison", "smartphone usage trends", "tech startups landscape", "film industry innovations", "popular gaming consoles", "social media evolution", "cybersecurity best practices", "virtual reality applications", "podcast popularity surge", "e-reader versus books", "music streaming platforms", "online education resources", "satellite versus cable tv", "mobile payment adoption", "ai in daily life", "cloud storage solutions", "5g network implications", "wearable tech advancements", "data privacy concerns", "youtube creators influence"],
        'Politics': ["electoral systems comparison", "health care policy reforms", "climate change legislation challenges", "immigration and border control", "political campaign financing regulations", "national security strategies", "voter turnout trends", "supranational organizations impact", "judiciary independence concerns", "lobbying ethics and rules", "intelligence oversight and ethics", "political party evolution", "foreign policy philosophies", "defense spending debates", "election technology and security", "public protest laws", "political satire influence", "constitutional amendment processes", "legislative gridlock causes", "diplomatic protocol differences"],
        'Social Dynamics': ["etiquette in public spaces", "attitudes toward immigration", "socializing in work environments", "education systems' social impact", "subcultures and identity formation", "social norms around drinking", "public transport interactions", "regional accents and perceptions", "community volunteering practices", "urban vs. rural lifestyles", "social implications of accents", "intergenerational relationship dynamics", "neighborhood watch schemes", "cultural approaches to aging", "holiday celebrations and traditions", "sports fandom social effects", "social etiquette in queuing", "cross-cultural marriage dynamics", "local food movements", "art scene community engagement"]
    }
    
    # questions based on topics
    question_prompt = """Please generate questions meeting the following criteria:
1. The answers must be based on objective facts (e.g., "What is the color of a football?") rather than subjective opinions (e.g., "Do you think drinking alcohol is morally acceptable?").
2. Each question should prompt two different answers depending on whether the respondent is an American English speaker or a British English speaker.
3. Each question should pertain specifically to the domain of {domain} and the topic of {topic}, but should exclude topics related to {other_domains}.

Here is a general example:
Question: What is the color of a football?
Answer for American English speakers: Brown.
Answer for British English speakers: Black and white.

Please generate {num_questions} such questions, ensuring they vary in structure and complexity. Do not provide answers for these questions. List each question as a separate line in your response.
"""

class StylishQuestionGen:
    system_prompt = """You are an expert in identifying subtle differences between American and British cultures, particularly in terms of word choices across various domains."""
    
    
    prompt = """Please rephrase or provide a prior 1~2-sentence context for the following question by incorporating 3 words from the provided list. These words are commonly used in British English, but are less used in American English. Please ensure the reference to British English usage is implicit, not explicit.

Word list:
```
{word_list_uk}
```

Original question: {question}

Rephrased question with 1-2 sentence additional context (which includes 3 of the words in the "Word list" above, while preserving the semantics of the "Original question"):

Format your answer as:
- Rephrased question: 
- Which of the 3 words are used:
"""

    def process_response(response, word_list_uk):
        from helper import check_words
        markers = ["- Rephrased question:", "- Which of the 3 words are used:"]
        if markers[0] in response and markers[1] in response:
            return re.sub(markers[0], "", re.split(markers[1], response)[0]).strip(), check_words(re.split(markers[1], response)[1], word_list_uk)
        else:
            return "", []


class AnswerGen:
    
    prompt = """{question}

Try to answer concisely with the key phrase, ideally in 10 words."""

    cot_prompt_1 = """{question}

Follow the steps below to answer the user's question above:
1. Infer the user's background based on the context and details provided in the question if possible. Consider factors such as the user's likely profession, level of expertise, geographical location, and any cultural nuances that may be relevant. 
2. Using this inferred background, answer the above question in a way that is tailored to the user's specific needs and context.

The output format should be like this:
1. <inferred user background>
2. <answer to the question>
Try to answer concisely with the key phrase, ideally in 20 words in total."""

    cot_prompt_2 = """In the following, you will see a question, and you need to answer it concisely.

Follow the following format:
- Question: {question}
- US or UK: (From the question, can you infer whether the user is from the US or UK?)
- Final answer: (Given the user's country, can you answer the question specific to their background?)

Now we begin.

- Question: {question}
- US or UK:"""

class ScoreGen:
    prompt = """
On a scale between 1 and 5, how similar are the following two
sentences? Respond only with a score between 1 and 5.

Input:
Sentence 1: Not really, too busy for that.
Sentence 2: Yes, quite fond of academic journals.
Output: 1
Input:
Sentence 1: Universal healthcare, accessible to all residents.
Sentence 2: NHS provides universal healthcare for all.
Output: 2
Input:
Sentence 1: Based on recommendations and personal interests.
Sentence 2: By author, genre, recommendations, and reviews.
Output: 3
Input:
Sentence 1: As often as I can.
Sentence 2: Quite often, I'm always willing.
Output: 4
Input:
Sentence 1: Yes, I love wearing hats!
Sentence 2: Yes, I quite fancy wearing hats.
Output: 5

Input:
Sentence 1: {sentence1}
Sentence 2: {sentence2}.
Output:"""

    def process_response(response):
        response = response.strip()
        try:
            # Try converting the string to float
            response = int(response)
            if response in [1,2,3,4,5]:
                return response
            else:
                return -1
        except:
            # Return None if the conversion fails
            return -1
