import json
import datasets
import pathlib
import re
import sqlite3
import hashlib
from lingua import LanguageDetectorBuilder


def process_txt_directory(dir_path, project_name, synthetic, mistakes):
    data = []
    for file in pathlib.Path(dir_path).iterdir():
        if file.suffix != '.txt':
            continue
        with open(file, 'r') as f:
            content = f.read()
        if content.startswith('§u§') and ('§a§' in content) and (content.count('§u§') == content.count('§a§')):
            sections = re.findall(r'§u§(.*?)§a§(.*?)(?=§u§|$)', content, re.DOTALL)
            data.append({'messages': [], 'project': project_name, 'synthetic': synthetic, 'mistakes': mistakes})
            for user_content, assistant_content in sections:
                data[-1]['messages'].append({'role': 'user', 'content': user_content.strip()})
                data[-1]['messages'].append({'role': 'assistant', 'content': assistant_content.strip()})
    return data

def process_moral_directory(dir_path, project_name, synthetic, mistakes):
    data = []
    for file in pathlib.Path(dir_path).iterdir():
        if file.suffix != '.txt':
            continue
        with open(file, 'r') as f:
            content = f.read()
        if '<|user_start|>' in content and '<|assistant_start|>' in content:
            # Extract user and assistant messages using the moral format
            user_matches = re.findall(r'<\|user_start\|>(.*?)<\|user_end\|>', content, re.DOTALL)
            assistant_matches = re.findall(r'<\|assistant_start\|>(.*?)<\|assistant_end\|>', content, re.DOTALL)

            # Pair user and assistant messages
            messages = []
            for i in range(min(len(user_matches), len(assistant_matches))):
                messages.append({'role': 'user', 'content': user_matches[i].strip()})
                messages.append({'role': 'assistant', 'content': assistant_matches[i].strip()})

            if messages:
                data.append({'messages': messages, 'project': project_name, 'synthetic': synthetic, 'mistakes': mistakes})
    return data


def process_moral_sqlite(db_path, project_name, synthetic, mistakes):
    data = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT prompt, response FROM messages")
    rows = cursor.fetchall()
    conn.close()
    for prompt, response in rows:
        messages = [{"role": "user", "content": prompt}, {'role': 'assistant', 'content': response}]
        data.append({'messages': messages, 'project': project_name, 'synthetic': synthetic, 'mistakes': mistakes})
    return data


def process_jsonl(file_path, project_name, synthetic, mistakes):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line_data = json.loads(line)
            line_data['project'] = project_name
            line_data['synthetic'] = synthetic
            line_data['mistakes'] = mistakes
            data.append(line_data)
    return data


def process_jsonl_ingredient_scanner(file_path, project_name, synthetic, mistakes):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line_data = json.loads(line)
            text = line_data['text']
            data.append({'messages': [{'role': 'user', 'content': text.split('\n<|im_end|>\n<|im_start|>assistant\n')[0].replace('<|im_start|>user\n', '')}, {'role': 'user', 'content': text.split('\n<|im_end|>\n<|im_start|>assistant\n')[1].replace('\n<|im_end|>', '')}], 'project': project_name, 'synthetic': synthetic, 'mistakes': mistakes})
    return data


def main():
    data_human_edited_misc = process_txt_directory('./human_edited/misc', 'other', False, False)
    data_human_edited_moral = process_moral_directory('./human_edited/moral', 'moral', False, False)
    data_synthetic_ingredient_scanner = process_jsonl_ingredient_scanner('./synthetic/ingredient_scanner/ingredient_scanner.jsonl', 'ingredient_scanner', True, False)
    data_synthetic_misc = process_txt_directory('./synthetic/misc', 'other', True, True)
    data_synthetic_moral = process_moral_sqlite('./synthetic/moral/database.sqlite3', 'moral', True, True)
    data_synthetic_topic_categorizer = process_jsonl('./synthetic/topic_categorizer/topic_categorizer.jsonl', 'topic_categorizer', True, False)

    data = data_human_edited_misc + data_human_edited_moral + data_synthetic_ingredient_scanner + data_synthetic_misc + data_synthetic_moral + data_synthetic_topic_categorizer

    detector = LanguageDetectorBuilder.from_all_spoken_languages().with_minimum_relative_distance(0.8).build()
    full_texts = [' '.join(msg['content'] for msg in item['messages']) for item in data]
    hashes = [hashlib.sha224(text.encode('utf-8'), usedforsecurity=False).hexdigest() for text in full_texts]

    conn = sqlite3.connect('language_cache.db')
    cursor = conn.cursor()

    placeholders = ','.join('?' for _ in hashes)
    cursor.execute(f"SELECT hash, languages FROM cache WHERE hash IN ({placeholders})", hashes)
    cached = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    to_detect = []
    indices_to_detect = []
    for i, h in enumerate(hashes):
        if h not in cached:
            to_detect.append(full_texts[i])
            indices_to_detect.append(i)

    if to_detect:
        results_list = detector.detect_multiple_languages_in_parallel_of(to_detect)
        for j, results in enumerate(results_list):
            idx = indices_to_detect[j]
            languages = list(set(result.language.iso_code_639_1.name for result in results))
            data[idx]['languages'] = languages  # noqa
            cursor.execute("INSERT OR REPLACE INTO cache (hash, languages) VALUES (?, ?)", (hashes[idx], json.dumps(languages)))

    for i, h in enumerate(hashes):
        if h in cached:
            data[i]['languages'] = cached[h]

    conn.commit()

    dataset = datasets.Dataset.from_list(data)
    dataset.to_parquet('data.parquet')


if __name__ == "__main__":
    main()
