import os
import re
import csv

email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
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

def find_pii(file_path, first_names, last_names, false_positives):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return [f"Error reading file: {str(e)}"]
    pii_found = []
    emails = re.findall(email_pattern, content, re.IGNORECASE)
    filtered_emails = [email for email in emails if email not in false_positives]
    if filtered_emails:
        pii_found.append('Emails: ' + ', '.join(set(filtered_emails)))
    phones = re.findall(phone_pattern, content)
    filtered_phones = [phone for phone in phones if phone not in false_positives]
    if filtered_phones:
        pii_found.append('Phone Numbers: ' + ', '.join(set(filtered_phones)))
    full_names = re.findall(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b', content)
    names_in_file = []
    for first, last in full_names:
        full_name = f"{first} {last}"
        if first in first_names and last in last_names and full_name not in false_positives:
            names_in_file.append(full_name)
    if names_in_file:
        pii_found.append('Full Names: ' + ', '.join(set(names_in_file)))  # unique
    return pii_found

def main():
    print("Loading name databases...")
    first_names, last_names = load_names()
    print(f"Loaded {len(first_names)} first names and {len(last_names)} last names.")
    print("Loading false positives...")
    false_positives = load_false_positives()
    print(f"Loaded {len(false_positives)} false positives.")
    pii_files = []
    for root, dirs, files in os.walk('.'):
        # Skip .venv and .idea directories
        if '.venv' in root or '.idea' in root:
            continue
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                pii = find_pii(path, first_names, last_names, false_positives)
                if pii:
                    pii_files.append((path, pii))
    if pii_files:
        print("Files containing potential PII:")
        for path, pii_types in pii_files:
            print(f"{path}: {', '.join(pii_types)}")
    else:
        print("No PII found in .txt files.")

if __name__ == "__main__":
    main()
