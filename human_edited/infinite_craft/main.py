import json

with open('prompt.md', 'r') as _f:
    PROMPT = _f.read()
with open('data.txt', 'r') as _f:
    DATA = _f.read().split('\n')


def main():
    output = []
    for line in DATA:
        if line:
            one = line.split('+')[0].strip()
            two = line.split('+')[1].split('=')[0].strip()
            out = line.split('=')[1].strip()
            for i in range(2):
                output.append({
                    'messages': [
                        {
                            'role': 'user',
                            'content': PROMPT.replace('{one}', one if i == 0 else two).replace('{two}', two if i == 0 else one)
                        },
                        {
                            'role': 'assistant',
                            'content': out
                        }
                    ]
                })
    with open('infinite_craft.jsonl', 'w') as _f:
        for line in output:
            _f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
