import json
import os


with open('tools.md', 'r') as _f:
    TOOLS = _f.read()


def main():
    if os.path.exists('cline.jsonl'):
        os.remove('cline.jsonl')
    for file in os.listdir('data'):
        if file.endswith('.json'):
            with open(os.path.join('data', file), 'r') as f:
                new_messages = []
                messages = json.load(f)
                for message in messages:
                    new_messages.append({
                        'role': message['role'],
                        'content': ''
                    })
                    new_messages[-1]['content'] = '\n\n'.join(i['text'] for i in message['content'] if i['type'] == 'text').split('</thinking>')[-1].strip()
                new_messages[0]['content'] = TOOLS + '\n\n' + new_messages[0]['content']
            with open('cline.jsonl', 'a') as f:
                f.write(json.dumps({'messages': new_messages}, ensure_ascii=False) + '\n')



if __name__ == '__main__':
    main()
