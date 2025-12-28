import sys
from util import db_read, PROMPT, MAX_DIFFERENCE
from pathlib import Path
from json import load as json_load, dumps as json_dumps
from tqdm import tqdm


def main() -> None:
    """
    Parse `data.txt` into a jsonl file which unsloth can understand
    :return: None
    """
    with open('data.txt', 'r') as f:
        file = f.read()
    sentences = [i.strip() for i in file.split('\n') if len(i.strip()) > 0]
    with open('topic_categorizer.jsonl', 'w', encoding='utf-8') as out_file:
        for i in tqdm(range(0, len(sentences) - 3, 1)):
            if i == 0:
                chunk = ['EMPTY'] + sentences[:4]
            elif i + 3 >= len(sentences):
                chunk = sentences[-5:-1] + ['EMPTY']
            else:
                chunk = sentences[i - 1:i + 4]
            db_results = db_read([v.split(' #')[0] for v in chunk if v != 'EMPTY'])
            if len(db_results['ids'][0]) == 0:
                continue
            topic_ids = []
            for j, result in enumerate(db_results['ids'][0]):
                if db_results['distances'][0][j] < MAX_DIFFERENCE:
                    id_ = result.split('-')[0]
                    if id_ not in topic_ids:
                        topic_ids.append(id_)
            if len(topic_ids) == 0:
                continue
            if len(topic_ids) == 1 and topic_ids[0] != '0':
                topic_ids.append('0')
            correct_topics = []
            for piece in chunk:
                if ' # ' in piece:
                    correct_topics.extend(piece.split(' # ')[1].split(' | '))
            correct_topics = list(set(topic.strip() for topic in correct_topics if topic.strip()))
            topics = []
            titles = {}
            loaded_topics = set()
            for topic_id in topic_ids:
                try:
                    with open(Path(__file__).resolve().parent / "parsed" / f"{topic_id}.json", 'r') as f:
                        topic_data = json_load(f)
                        if topic_data['topic'] not in loaded_topics:
                            topics.append(topic_data)
                            titles[topic_data['topic']] = len(topics) - 1
                            loaded_topics.add(topic_data['topic'])
                except FileNotFoundError:
                    continue
            for topic in correct_topics:
                if topic not in loaded_topics:
                    for topic_file in (Path(__file__).resolve().parent / "parsed").glob("*.json"):
                        if topic_file.name == 'hashes.json':
                            continue
                        with open(topic_file, 'r') as f:
                            topic_data = json_load(f)
                            if topic_data['topic'] == topic and topic_data['topic'] not in loaded_topics:
                                topics.append(topic_data)
                                titles[topic_data['topic']] = len(topics) - 1
                                loaded_topics.add(topic_data['topic'])
                                break
            formatted_topics = ''
            topics.sort(key=lambda x: x['topic'])
            for topic in topics:
                if len(formatted_topics) > 0:
                    formatted_topics += '\n'
                formatted_topics += f"'{topic['topic']}'"
            user = PROMPT.replace('{TOPICS}', formatted_topics)
            for j, sentence in enumerate(chunk):
                user = user.replace('{' + f"SENTENCE_{j+1}" + '}', sentence.split(' #')[0])
            assistant = f"```python\n[{', '.join(["'" + v + "'" for v in correct_topics])}]\n```"
            line = {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}
            line = json_dumps(line)
            out_file.write(line + '\n')
            if i + 3 >= len(sentences):
                break


if __name__ == '__main__':
    main()
