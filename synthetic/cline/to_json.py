import json
import os


with open('tools.md', 'r') as _f:
    TOOLS = _f.read()


def tool_use_to_xml(tool_use):
    """Convert a tool_use content item to its XML string representation."""
    name = tool_use['name']
    params = tool_use['input']
    
    # For plan_mode_respond, only output the response parameter
    if name == 'plan_mode_respond':
        params_to_output = {'response': params.get('response', '')}
    else:
        params_to_output = params
    
    xml_parts = [f'<{name}>']
    for key, value in params_to_output.items():
        if '\n' in str(value):
            xml_parts.append(f'<{key}>\n{value}\n</{key}>')
        else:
            xml_parts.append(f'<{key}>{value}</{key}>')
    xml_parts.append(f'</{name}>')
    
    return '\n'.join(xml_parts)


def main():
    if os.path.exists('cline.jsonl'):
        os.remove('cline.jsonl')
    for file in os.listdir('data'):
        if file.endswith('.json'):
            with open(os.path.join('data', file), 'r') as f:
                new_messages = [{
                    'role': 'system',
                    'content': TOOLS
                }]
                messages = json.load(f)
                for message in messages:
                    new_messages.append({
                        'role': message['role'],
                        'content': ''
                    })
                    contents = []
                    for item in message['content']:
                        if item['type'] == 'text' and item['text'] != '\nthought\n':
                            contents.append(item['text'])
                        elif item['type'] == 'thinking':
                            contents.append(f"<thinking>{item['thinking']}</thinking>")
                        elif item['type'] == 'tool_use':
                            contents.append(tool_use_to_xml(item))
                    new_messages[-1]['content'] = '\n\n'.join(contents)
            with open('cline.jsonl', 'a') as f:
                f.write(json.dumps({'messages': new_messages}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
