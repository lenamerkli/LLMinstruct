import sys
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


def main():
    llm = get_llm(MODEL)
    llm.load_model()
    while llm.is_loading() or not llm.is_running():
        sleep(1)
    conversation = []
    board = ChessBoard()
    try:
        loop1 = True
        while loop1:
            # take user input
            # add user input to conversation
            loop2 = True
            while loop2:
                # call llm
                # add llm response to conversation
                # extract tool calls
                if tool_calls:
                    # run tool calls
                    # the tool calls all will check for losses
                    # add the tool response(s) to conversation
                    if loss:
                        loop1 = False
                else:
                    loop2 = False
    finally:
        llm.stop()
