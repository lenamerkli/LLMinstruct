import os
import certifi
import json
import datasets
import pathlib
import re
import sqlite3
import csv
import datetime
import subprocess
import lingua
import transformers
import typing as t
import tqdm

from language_cache import LanguageCache


os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()


EMAIL_PATTERN = r'\b[\wÀ-ÿ0-9._%+-]+@[\wÀ-ÿ0-9.-]+\.[\wÀ-ÿ]{2,}\b'
PHONE_PATTERN = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
LANGUAGES_TO_DETECT = [
    lingua.Language.GERMAN,
    lingua.Language.ENGLISH,
    lingua.Language.FRENCH,
    lingua.Language.ITALIAN,
    lingua.Language.CHINESE,
    lingua.Language.RUSSIAN,
]

# Page number offsets for OCR documents (doc_name -> offset to apply to page numbers)
PAGE_OFFSETS = {
    "der-taumelnde-kontinent": -1,  # page N → file N-1
    "wilhelm-tell": +4,               # page N → file N+4
}


def count_tokens(messages: t.List[t.Dict], tokenizer, template_env=None) -> int:
    encoded_input = tokenizer.apply_chat_template(messages, tokenize=True, template_env=template_env)
    return len(encoded_input)


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


def load_false_negatives():
    false_negatives = set()
    with open('false_negatives.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                false_negatives.add(line)
    return false_negatives


def get_git_tracked_files(dir_path):
    """Get list of files tracked by git in the specified directory."""
    try:
        result = subprocess.run(
            ['git', 'ls-files', str(dir_path)],
            capture_output=True,
            text=True,
            check=True
        )
        return set(result.stdout.splitlines())
    except subprocess.CalledProcessError:
        # If git command fails, return empty set (no files will be processed)
        return set()


def process_txt_directory(dir_path, project_name, synthetic, mistakes):
    data = []
    git_files = get_git_tracked_files(dir_path)
    for file in pathlib.Path(dir_path).iterdir():
        if file.suffix != '.txt':
            continue
        if str(file) not in git_files:
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
        if isinstance(response, str) and len(response) > 2:
            messages = [{"role": "user", "content": prompt}, {'role': 'assistant', 'content': response}]
            data.append({'messages': messages, 'project': project_name, 'synthetic': synthetic, 'mistakes': mistakes})
    return data


def process_biasbench_sqlite(db_path, project_name, synthetic, mistakes):
    data = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT prompt, response FROM responses")  # noqa
    rows = cursor.fetchall()
    conn.close()
    for prompt, response in rows:
        if isinstance(response, str) and len(response) > 2:
            messages = json.loads(prompt)
            messages.append({'role': 'assistant', 'content': response})
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


def process_attachments(message):
    """Process attachments in message content and add them to the attachments list."""
    import re
    import os

    # Pattern to match <attach:{{hash}}> where hash is a 64-character hex string (SHA256)
    attachment_pattern = r'<attach:([a-f0-9]{64})>'

    # Find all attachment patterns in the content
    matches = re.findall(attachment_pattern, message.get('content', ''))

    # Process each attachment
    for hash_value in matches:
        # Check if file exists in attachments/data directory
        file_path = os.path.join('attachments', 'data', hash_value)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Attachment file not found: {file_path} (hash: {hash_value})")

        # Add attachment to the message
        if 'attachments' not in message:
            message['attachments'] = []
        mime_type = subprocess.run(['file', '--mime-type', '-b', file_path], capture_output=True, text=True).stdout.strip()
        if not mime_type:
            mime_type = 'application/octet-stream'
        message['attachments'].append({'type': f"{mime_type}/sha256sum", 'value': f'sha256sum:{hash_value}'})
    # Remove all attachment patterns from content
    if matches:
        message['content'] = re.sub(attachment_pattern, '', message.get('content', ''))

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
        for message in messages:
            if '</think>' in message['content']:
                message['content'] = message['content'].split('</think>')[1]
            message['content'] = re.sub(r"<tool_call>.*?</tool_call>", '', message['content'], flags=re.DOTALL).strip()
            if 'tool_calls' in message:
                tool_calls = message['tool_calls']
                del message['tool_calls']
                if len(tool_calls) > 0:
                    if 'attachments' not in message:
                        message['attachments'] = []
                    for tool_call in tool_calls:
                        message['attachments'].append({'type': 'application/json/tool_call', 'value': json.dumps(tool_call)})
        if len(messages) > 1:
            messages[0]['attachments'] = [{'type': 'application/json/tools', 'value': json.dumps(tools)}]
            data.append({'messages': messages, 'project': project_name, 'synthetic': synthetic, 'mistakes': mistakes})
            messages2 = [i for i in messages if ('<tool_call>' not in i['content']) and (i['role'] not in ['tool', 'system'])]
            data.append({'messages': messages2, 'project': project_name, 'synthetic': synthetic, 'mistakes': mistakes})
    return data

def process_explain_meme(dir_path, project_name, synthetic, mistakes):
    with open(pathlib.Path('./attachments/rename_log.txt')) as f:
        rename_log = f.read()
    data = []
    conn = sqlite3.connect((dir_path + '/database.sqlite3').replace('//', '/'))
    cursor = conn.cursor()
    cursor.execute("SELECT meme, response FROM responses")
    rows = cursor.fetchall()
    conn.close()
    for meme, response in rows:
        if isinstance(response, str) and len(response) > 2:
            hash_ = None
            for line in rename_log.split('\n'):
                if line.startswith(meme + ' -> '):
                    hash_ = line.split(' -> ')[1]
                    break
            if hash_:
                messages = [{'role': 'user', 'content': f"<attach:{hash_}>Explain me this meme."}, {'role': 'assistant', 'content': response}]
                data.append({'messages': messages, 'project': project_name, 'synthetic': synthetic, 'mistakes': mistakes})
    return data


def process_ocr(dir_path, project_name, synthetic, mistakes):
    """Process OCR data: images + text files with page markers."""
    with open(pathlib.Path('./attachments/rename_log.txt')) as f:
        rename_log = f.read()

    # Build mapping from image path to hash
    image_to_hash = {}
    for line in rename_log.split('\n'):
        if ' -> ' in line:
            original, hash_value = line.split(' -> ', 1)
            image_to_hash[original.strip()] = hash_value.strip()

    # Read the OCR prompt
    with open(pathlib.Path(dir_path) / 'prompt.md', 'r') as f:
        ocr_prompt = f.read().strip()

    # Read all text files
    texts_dir = pathlib.Path(dir_path) / 'texts'
    images_dir = pathlib.Path(dir_path) / 'images'

    data = []
    for text_file in texts_dir.iterdir():
        if text_file.suffix != '.txt':
            continue

        with open(text_file, 'r') as f:
            content = f.read()

        # Extract document name from text file name (without extension)
        doc_name = text_file.stem

        # Parse pages from text content using <page n="N"> tags
        pages = re.findall(r'<page n="(\d+)">\s*(.*?)\s*</page>', content, re.DOTALL)

        # Get corresponding image directory
        image_dir = images_dir / doc_name
        if not image_dir.exists():
            continue

        # Get list of image files sorted by page number
        image_files = sorted(image_dir.glob('*.png'))

        for page_num, page_text in pages:
            page_num_int = int(page_num)
            # Apply document-specific offset if defined
            offset = PAGE_OFFSETS.get(doc_name, 0)
            # Find corresponding image (page_0001.png for page 1, etc.)
            image_file = image_dir / f'page_{page_num_int + offset:04d}.png'
            if not image_file.exists():
                continue

            # Find hash for this image
            rel_image_path = str(image_file.relative_to(images_dir))
            if rel_image_path not in image_to_hash:
                continue

            hash_value = image_to_hash[rel_image_path]

            # Create conversation for this page
            user_content = f"<attach:{hash_value}>{ocr_prompt}"
            messages = [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': page_text.strip()}
            ]
            data.append({'messages': messages, 'project': project_name, 'synthetic': synthetic, 'mistakes': mistakes})

    return data


def check_pii_in_data(data, first_names, last_names, false_positives, false_negatives):
    pii_findings = []
    for idx, entry in tqdm.tqdm(enumerate(data), desc='Checking for PII', total=len(data)):
        content_parts = []
        for message in entry.get('messages', []):
            content_parts.append(message.get('content', ''))
        full_content = ' '.join(content_parts).replace(' ', ' ')
        while '  ' in full_content:
            full_content = full_content.replace('  ', ' ')
        emails = re.findall(EMAIL_PATTERN, full_content, re.IGNORECASE)
        filtered_emails = [email for email in emails if email not in false_positives]
        phones = re.findall(PHONE_PATTERN, full_content)
        filtered_phones = [phone for phone in phones if phone not in false_positives]
        full_names = re.findall(r'\b([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)\s+([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)\b', full_content)
        names_in_content = []
        for first, last in full_names:
            full_name = f"{first} {last}"
            if first in first_names and last in last_names and full_name not in false_positives:
                names_in_content.append(full_name)
        for full_name in false_negatives:
            for name_in_content in names_in_content:
                if name_in_content in full_name:
                    while name_in_content in names_in_content:
                        names_in_content.remove(name_in_content)
            if full_name in full_content:
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


def strftime_now(format_str):
    return datetime.datetime.now().strftime(format_str)


def main():
    data_human_edited_biasbench = process_txt_directory('./human_edited/biasbench', 'biasbench', False, False)
    data_human_edited_infinite_craft = process_jsonl('./human_edited/infinite_craft/infinite_craft.jsonl', 'infinite_craft', False, False)
    data_human_edited_misc = process_txt_directory('./human_edited/misc', 'misc', False, False)
    data_human_edited_moral = process_moral_directory('./human_edited/moral', 'moral', False, False)
    data_synthetic_biasbench = process_biasbench_sqlite('./synthetic/biasbench.sqlite3', 'biasbench', True, True)
    data_synthetic_drawback_chess = process_drawback_chess_directory('./synthetic/drawback_chess/conversations', 'drawback_chess', True, True)
    data_synthetic_explain_meme = process_explain_meme('./synthetic/explain_meme', 'explain_meme', True, True)
    data_synthetic_ingredient_scanner = process_jsonl_ingredient_scanner('./synthetic/ingredient_scanner/ingredient_scanner.jsonl', 'ingredient_scanner', True, False)
    data_synthetic_ingredient_scanner2 = process_jsonl('./synthetic/ingredient_scanner/ingredient_scanner2.jsonl', 'ingredient_scanner', False, True)
    data_synthetic_misc = process_txt_directory('./synthetic/misc', 'misc', True, True)
    data_synthetic_moral = process_moral_sqlite('./synthetic/moral/database.sqlite3', 'moral', True, True)
    data_synthetic_ocr = process_ocr('./synthetic/ocr', 'ocr', True, True)
    data_synthetic_topic_categorizer = process_jsonl('./synthetic/topic_categorizer/topic_categorizer.jsonl', 'topic_categorizer', True, False)

    data = data_human_edited_biasbench + data_human_edited_infinite_craft + data_human_edited_misc + data_human_edited_moral + data_synthetic_biasbench + data_synthetic_drawback_chess + data_synthetic_explain_meme + data_synthetic_ingredient_scanner + data_synthetic_ingredient_scanner2 + data_synthetic_misc + data_synthetic_moral + data_synthetic_ocr + data_synthetic_topic_categorizer

    first_names, last_names = load_names()
    false_positives = load_false_positives()
    false_negatives = load_false_negatives()
    pii_findings = check_pii_in_data(data, first_names, last_names, false_positives, false_negatives)

    # Collect all unique PII
    all_emails = set()
    all_phones = set()
    all_names = set()
    for finding in pii_findings:
        for pii_type in finding['pii_found']:
            if pii_type.startswith('Emails: '):
                emails_str = pii_type[len('Emails: '):]
                emails = [e.strip() for e in emails_str.split(', ')]
                all_emails.update(emails)
            elif pii_type.startswith('Phone Numbers: '):
                phones_str = pii_type[len('Phone Numbers: '):]
                phones = [p.strip() for p in phones_str.split(', ')]
                all_phones.update(phones)
            elif pii_type.startswith('Full Names: '):
                names_str = pii_type[len('Full Names: '):]
                names = [n.strip() for n in names_str.split(', ')]
                all_names.update(names)

    with open('pii_report.txt', 'w', encoding='utf-8') as f:
        f.write("Personally Identifiable Information (PII) Report\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
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

        # Add deduplicated PII section
        f.write("\nAll PII Found (Deduplicated):\n")
        if all_emails:
            f.write(f"- Emails: {', '.join(sorted(all_emails))}\n")
        else:
            f.write("- Emails: None\n")
        if all_phones:
            f.write(f"- Phone Numbers: {', '.join(sorted(all_phones))}\n")
        else:
            f.write("- Phone Numbers: None\n")
        if all_names:
            f.write(f"- Full Names: {', '.join(sorted(all_names))}\n")
        else:
            f.write("- Full Names: None\n")

    detector = lingua.LanguageDetectorBuilder.from_languages(*LANGUAGES_TO_DETECT).with_minimum_relative_distance(0.8).build()
    full_texts = [' '.join(msg['content'] for msg in item['messages']) for item in data]

    with LanguageCache('language_cache.db') as cache:
        languages_list = cache.detect_with_cache(detector, full_texts)
    for i, languages in enumerate(languages_list):
        data[i]['languages'] = languages

    for i in data:
        for j in i['messages']:
            if 'attachments' not in j:
                j['attachments'] = []

    # Process attachments in all messages
    for entry in tqdm.tqdm(data, desc='Processing attachments'):
        for message in entry['messages']:
            process_attachments(message)

    tokenizer = transformers.AutoTokenizer.from_pretrained('./tokenizer')
    for i, element in tqdm.tqdm(enumerate(data), desc='Counting tokens', total=len(data)):
        try:
            template_env = {'strftime_now': strftime_now}
            messages = [{k: v for k, v in msg.items() if k not in ['attachments', 'tool_calls']} for msg in element['messages']]
            data[i]['token_count'] = count_tokens(messages, tokenizer, template_env)
        except Exception as e:
            e.add_note(f"{i}\n{element['messages']}")
            raise e

    for entry in data:
        entry['tool_calling'] = False
        for message in entry['messages']:
            for attachment in message['attachments']:
                if attachment['type'] in ['application/json/tools', 'application/json/tool_call']:
                    entry['tool_calling'] = True
                    break
            if entry['tool_calling']:
                break

    for entry in data:
        entry['vision'] = False
        for message in entry['messages']:
            for attachment in message['attachments']:
                if attachment['type'].startswith('image/') or attachment['type'].startswith('application/pdf'):
                    entry['vision'] = True
                    break
            if entry['vision']:
                break

    dataset = datasets.Dataset.from_list(data)
    dataset.to_parquet('data.parquet')


if __name__ == "__main__":
    main()
