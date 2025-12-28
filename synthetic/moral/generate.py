import sys
import pathlib
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from database import query_db
from llm import get_llm
from time import sleep
from dotenv import load_dotenv


load_dotenv(dotenv_path=str(project_root / '.env'))


APERTUS_MODEL = 'publicai/swiss-ai/apertus-70b-instruct'
TRANSLATION_MODEL = 'NVIDIA-Nemotron-Nano-9B-v2-Q5_K_S.gguf'
LANGUAGES = ['en', 'de', 'fr']


def translate(question, language, llm):
    conversation = [
        {'role': 'user', 'content': f"Translate the following question into `{language}`:\n```text\n{question}\n```\nReturn your answer inside a pair of three backticks (```), e.g.```text\nTranslation\n```"}
    ]
    response = llm.generate(conversation)
    if '```' in response:
        parts = response.split('```')
        if len(parts) >= 3:
            content = parts[1].strip()
            if '\n' in content:
                content = content.split('\n', 1)[1].strip()
            return content
    return response.strip()

def all_translations_cached(questions):
    for question in questions:
        for language in LANGUAGES:
            if language == 'en':
                continue
            existing_translation = query_db(
                "SELECT 1 FROM translations WHERE original = ? AND language = ? LIMIT 1",
                (question, language),
                one=True
            )
            if not existing_translation:
                return False
    return True


def main():
    with open('questions.txt', 'r') as f:
        questions = [line.strip() for line in f]
    existing = query_db("SELECT prompt FROM messages")
    existing_prompts = {row[0] for row in existing}
    llm = None
    if not all_translations_cached(questions):
        llm = get_llm(TRANSLATION_MODEL)
        llm.load_model()
        while llm.is_loading() or not llm.is_running():
            sleep(1)
    for question in questions:
        if question not in existing_prompts:
            query_db("INSERT INTO messages (prompt, response) VALUES (?, ?)", (question, ""))
        for language in LANGUAGES:
            if language == 'en':
                continue
            existing_translation = query_db(
                "SELECT 1 FROM translations WHERE original = ? AND language = ? LIMIT 1",
                (question, language),
                one=True
            )
            if existing_translation:
                continue
            if llm is not None:
                translation = translate(question, language, llm)
                query_db("INSERT INTO translations (original, translation, language) VALUES (?, ?, ?)", (question, translation, language))
    if llm is not None:
        llm.stop()
    translations = {row[1] for row in query_db("SELECT * FROM translations")}
    existing_prompts = {row[0] for row in existing}
    for translation in translations:
        if translation not in existing_prompts:
            query_db("INSERT INTO messages (prompt, response) VALUES (?, ?)", (translation, ""))
    unanswered_prompts = {row[0] for row in query_db("SELECT * FROM messages WHERE response = '' OR response IS NULL")}
    llm = get_llm(APERTUS_MODEL)
    llm.load_model()
    while llm.is_loading() or not llm.is_running():
        sleep(1)
    with open('system.md', 'r') as f:
        system_prompt = f.read()
    for prompt in unanswered_prompts:
        print('---\n\n' + prompt + '\n\n')
        conversation = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
        response = llm.generate(conversation)
        print(response + '\n\n')
        query_db("UPDATE messages SET response = ? WHERE prompt = ?", (response, prompt))
    llm.stop()


if __name__ == '__main__':
    main()
