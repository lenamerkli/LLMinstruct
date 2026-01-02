import json
import datasets
import pathlib
import re
import sqlite3
import csv


email_pattern = r'\b[\wÀ-ÿ0-9._%+-]+@[\wÀ-ÿ0-9.-]+\.[\wÀ-ÿ]{2,}\b'
phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'

def load_names():
    first_names = set()
    with open('first_names.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            first_names.add(row[0])
    for element in ['Die', 'The', 'In']:
        if element in first_names:
            first_names.remove(element)
    last_names = set()
    with open('last_names.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            last_names.add(row[0])

    return first_names, last_names

def load_false_positives():
    false_positives = set()
    with open('false_positives.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                false_positives.add(line)
    return false_positives


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


def process_drawback_chess_directory(dir_path, project_name, synthetic, mistakes):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Makes the provided move on an internal state.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "move": {"type": "string",
                                 "description": "The move in SAN (e.g., e4, Nf3) or UCI (e.g., e2e4) format."}
                    },
                    "required": ["move"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "best",
                "description": "Calculates the best move for you. Only takes your drawback into account.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "board",
                "description": "Gets the current board in a fancy format.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]
    data = []
    for file in pathlib.Path(dir_path).iterdir():
        if file.suffix != '.json':
            continue
        with open(file, 'r') as f:
            content = f.read()
        messages = json.loads(content)
        data.append({'messages': messages, 'tools': tools, 'project': project_name, 'synthetic': synthetic, 'mistakes': mistakes})
    return data


def check_pii_in_data(data, first_names, last_names, false_positives):
    pii_findings = []
    for idx, entry in enumerate(data):
        content_parts = []
        for message in entry.get('messages', []):
            content_parts.append(message.get('content', ''))
        full_content = ' '.join(content_parts)
        emails = re.findall(email_pattern, full_content, re.IGNORECASE)
        filtered_emails = [email for email in emails if email not in false_positives]
        phones = re.findall(phone_pattern, full_content)
        filtered_phones = [phone for phone in phones if phone not in false_positives]
        full_names = re.findall(r'\b([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)\s+([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)\b', full_content)
        names_in_content = []
        for first, last in full_names:
            full_name = f"{first} {last}"
            if first in first_names and last in last_names and full_name not in false_positives:
                names_in_content.append(full_name)
        pii_types = []
        if filtered_emails:
            pii_types.append(f"Emails: {', '.join(set(filtered_emails))}")
        if filtered_phones:
            pii_types.append(f"Phone Numbers: {', '.join(set(filtered_phones))}")
        if names_in_content:
            pii_types.append(f"Full Names: {', '.join(set(names_in_content))}")
        if pii_types:
            finding = {
                'entry_index': idx,
                'project': entry.get('project', 'unknown'),
                'content': '\t'.join(content_parts).replace('\n', ' '),
                'pii_found': pii_types,
            }
            pii_findings.append(finding)
    return pii_findings


def main():
    data_human_edited_biasbench = process_txt_directory('./human_edited/biasbench', 'biasbench', False, False)
    data_human_edited_misc = process_txt_directory('./human_edited/misc', 'misc', False, False)
    data_human_edited_moral = process_moral_directory('./human_edited/moral', 'moral', False, False)
    data_synthetic_drawback_chess = process_drawback_chess_directory('./synthetic/drawback_chess/conversations', 'drawback_chess', True, True)
    data_synthetic_ingredient_scanner = process_jsonl_ingredient_scanner('./synthetic/ingredient_scanner/ingredient_scanner.jsonl', 'ingredient_scanner', True, False)
    data_synthetic_misc = process_txt_directory('./synthetic/misc', 'misc', True, True)
    data_synthetic_moral = process_moral_sqlite('./synthetic/moral/database.sqlite3', 'moral', True, True)
    data_synthetic_topic_categorizer = process_jsonl('./synthetic/topic_categorizer/topic_categorizer.jsonl', 'topic_categorizer', True, False)

    data = data_human_edited_biasbench + data_human_edited_misc + data_human_edited_moral + data_synthetic_drawback_chess + data_synthetic_ingredient_scanner + data_synthetic_misc + data_synthetic_moral + data_synthetic_topic_categorizer

    first_names, last_names = load_names()
    false_positives = load_false_positives()
    pii_findings = check_pii_in_data(data, first_names, last_names, false_positives)
    with open('pii_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"Total entries checked: {len(data)}\n")
        f.write(f"Entries with PII found: {len(pii_findings)}\n\n")
        if pii_findings:
            f.write("PII Findings:\n")
            f.write("-" * 40 + "\n\n")
            for finding in pii_findings:
                f.write(f"Entry Index: {finding['entry_index']}\n")
                f.write(f"Project: {finding['project']}\n")
                f.write(f"Content: {finding['content']}\n")
                f.write("PII Found:\n")
                for pii_type in finding['pii_found']:
                    f.write(f"  - {pii_type}\n")
                f.write("\n" + "-" * 40 + "\n\n")
        else:
            f.write("No PII found in any entries.\n")

    dataset = datasets.Dataset.from_list(data)
    dataset.to_parquet('data.parquet')


if __name__ == "__main__":
    main()
