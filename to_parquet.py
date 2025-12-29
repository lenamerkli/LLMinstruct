import json
import datasets
import pathlib
import re
import sqlite3


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
    data_human_edited_misc = process_txt_directory('./human_edited/misc', 'misc', False, False)
    data_human_edited_moral = process_moral_directory('./human_edited/moral', 'moral', False, False)
    data_synthetic_ingredient_scanner = process_jsonl_ingredient_scanner('./synthetic/ingredient_scanner/ingredient_scanner.jsonl', 'ingredient_scanner', True, False)
    data_synthetic_misc = process_txt_directory('./synthetic/misc', 'misc', True, True)
    data_synthetic_moral = process_moral_sqlite('./synthetic/moral/database.sqlite3', 'moral', True, True)
    data_synthetic_topic_categorizer = process_jsonl('./synthetic/topic_categorizer/topic_categorizer.jsonl', 'topic_categorizer', True, False)

    data = data_human_edited_misc + data_human_edited_moral + data_synthetic_ingredient_scanner + data_synthetic_misc + data_synthetic_moral + data_synthetic_topic_categorizer

    dataset = datasets.Dataset.from_list(data)
    dataset.to_parquet('data.parquet')


if __name__ == "__main__":
    main()
