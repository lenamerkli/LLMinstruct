"""
In data.json, there are ingredient and contaminant lists.
These correspond with the beginning of image names normalized in the attachments folder.
See rename_log.txt for their new file names.

This script generates question and answer pairs in the format that to_parquet.py internally uses.
Do not try to match the format of ingredient_scanner.jsonl.
A new loader in to_parquet.py is necessary.

The other generate.py is for something else.
"""

import json


QUESTION = 'Was sind die Zutaten und die Verunreinigungen dieses Produkts? Gib mir beide Listen zwischen zwei ```.'
RESPONSE_TEMPLATE = """
**Zutaten**
```text
{ingredients}
```

**Verunreinigungen**
```
{contaminants}
```
"""

def load_data():
    with open('data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_rename_log():
    mapping = {}
    with open('../../attachments/rename_log.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ' -> ' in line:
                original, hashed = line.split(' -> ')
                mapping[original] = hashed
    return mapping

def main():
    data = load_data()
    rename_mapping = load_rename_log()

    output = []
    for original_name, hashed in rename_mapping.items():
        # Extract product_id from original_name (e.g., '07429_0001.png' -> '07429')
        product_id = original_name.split('_')[0]

        if product_id not in data:
            # Skip files from other projects
            continue

        info = data[product_id]

        # Build question with single attachment
        question_content = QUESTION + f' <attach:{hashed}>'

        # Build response, replace empty lists with 'keine'
        ingredients = info.get('Zutaten', '').strip() or 'keine'
        contaminants = info.get('Verunreinigungen', '').strip() or 'keine'
        response_content = RESPONSE_TEMPLATE.format(
            ingredients=ingredients,
            contaminants=contaminants
        ).strip()

        # Create message pair
        messages = [
            {'role': 'user', 'content': question_content},
            {'role': 'assistant', 'content': response_content}
        ]

        output.append({'messages': messages})

    # Write to jsonl
    with open('ingredient_scanner2.jsonl', 'w', encoding='utf-8') as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
