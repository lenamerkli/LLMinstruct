import sys
import json
import re
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from llm import get_llm
from time import sleep
from sunfish import ChessBoard


MODEL = 'Qwen3-30B-A3B-Q5_K_M.gguf'
with open('prompt.md', 'r') as _f:
    PROMPT = _f.read()
# with open('drawbacks.txt', 'r') as _f:
#     DRAWBACKS = _f.read().strip().splitlines()
DRAWBACKS = ['**Lucky Bastard**: You do not have a drawback.']  # debug only
if MODEL.startswith('Apertus'):
    with open(Path(__file__).parent.parent.parent / 'chat_template.jinja') as _f:
        CHAT_TEMPLATE = _f.read()
elif MODEL.startswith('Qwen3'):
    with open(Path(__file__).parent.parent.parent / 'Qwen3.jinja') as _f:
        CHAT_TEMPLATE = _f.read()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Makes the provided move on an internal state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "move": {"type": "string", "description": "The move in SAN (e.g., e4, Nf3) or UCI (e.g., e2e4) format."}
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


def extract_tool_calls(content):
    tool_calls = []
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
    for match in matches:
        try:
            tool_calls.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return tool_calls


def main():
    llm = get_llm(MODEL)
    llm.load_model()
    while llm.is_loading() or not llm.is_running():
        sleep(1)

    board = ChessBoard()
    drawback = DRAWBACKS[0]  # Using the debug drawback
    system_prompt = PROMPT.format(
        drawback=drawback,
        color="Black",
        position=board.to_string()
    )

    conversation = [{"role": "system", "content": system_prompt}]

    try:
        loop1 = True
        while loop1:
            user_input = input("User: ")
            conversation.append({"role": "user", "content": user_input})

            loop2 = True
            while loop2:
                response = llm.generate(
                    conversation,
                    chat_template=CHAT_TEMPLATE,
                    template_env={'tools': TOOLS},
                    stop=['</tool_call>'],
                )

                if response.count('<tool_call>') == response.count('</tool_call>') + 1:
                    response += '</tool_call>'

                tool_calls = extract_tool_calls(response)
                # We need to add the assistant's response to the conversation.
                # If it contains tool calls, we should format it correctly for the template.
                assistant_msg = {"role": "assistant", "content": response}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in tool_calls
                    ]
                conversation.append(assistant_msg)

                if tool_calls:
                    for tc in tool_calls:
                        name = tc["name"]
                        args = tc["arguments"]
                        result = ""
                        loss = False

                        if name == "move":
                            try:
                                move_str = args["move"]
                                board.move(move_str)
                                result = f"Move {move_str} played successfully."
                                if board.check_loss():
                                    result += " The game is over."
                                    loss = True
                            except Exception as e:
                                result = f"Error: {str(e)}"
                        elif name == "best":
                            best_move = board.get_best_move()
                            result = best_move if best_move else "No legal moves found."
                        elif name == "board":
                            result = board.to_string()

                        conversation.append({"role": "tool", "content": result})
                        if loss:
                            loop1 = False
                            loop2 = False
                            break
                else:
                    # No tool calls, this is the final response for this turn
                    loop2 = False
    except EOFError:
        pass
    finally:
        llm.stop()
        conv_dir = Path(__file__).parent / 'conversations'
        conv_dir.mkdir(exist_ok=True)
        i = 0
        while (conv_dir / f"{i}.json").exists():
            i += 1
        with open(conv_dir / f"{i}.json", 'w') as f:
            json.dump(conversation, f, indent=4)


if __name__ == '__main__':
    main()
