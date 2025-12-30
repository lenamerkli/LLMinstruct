import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from llm import get_llm
from time import sleep
from random import randint
import json
import re
from sunfish import Position, Searcher, apply_move, visualize_position, initial


# MODEL = 'Apertus-8B-Instruct-2509-Q5_K_M.gguf'
# MODEL = 'Apertus-70B-Instruct-2509-Q4_K_M.gguf'
MODEL = 'Qwen3-30B-A3B-Q5_K_M.gguf'

STARTING_POSITION = """
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟ 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . . . . 3
2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 1
  a b c d e f g h
""".strip()
with open('prompt.md', 'r') as _f:
    PROMPT = _f.read()
with open('drawbacks.txt', 'r') as _f:
    DRAWBACKS = _f.read().strip().splitlines()
if MODEL.startswith('Apertus'):
    with open(Path(__file__).parent.parent.parent / 'chat_template.jinja') as _f:
        CHAT_TEMPLATE = _f.read()
elif MODEL.startswith('Qwen3'):
    with open(Path(__file__).parent.parent.parent / 'Qwen3.jinja') as _f:
        CHAT_TEMPLATE = _f.read()


def position_to_string(position):
    """Convert a Position object to the visual string format expected by the LLM."""
    return visualize_position(position)


def string_to_position(position_str):
    """Convert a visual position string back to a Position object."""
    # Unicode to sunfish piece mapping
    piece_map = {
        '♙': 'P', '♖': 'R', '♘': 'N', '♗': 'B', '♕': 'Q', '♔': 'K',
        '♟': 'p', '♜': 'r', '♞': 'n', '♝': 'b', '♛': 'q', '♚': 'k',
        '.': '.'
    }

    lines = position_str.strip().split('\n')

    # Find the board lines (skip file labels and rank numbers)
    board_lines = []
    for line in lines:
        line = line.strip()
        if len(line) >= 15 and line[1] == ' ' and line[0].isdigit():  # Rank line like "8 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 8"
            # Extract pieces (skip rank number and spaces)
            pieces = line.split()[1:-1]  # Skip first (rank) and last (rank) elements
            board_lines.append(pieces)

    if len(board_lines) != 8:
        raise ValueError(f"Expected 8 board lines, got {len(board_lines)}")

    # Build sunfish board (120 chars with padding)
    board = list('         \n' * 2)  # Top padding

    for rank_idx, pieces in enumerate(board_lines):
        if len(pieces) != 8:
            raise ValueError(f"Expected 8 pieces in rank {8-rank_idx}, got {len(pieces)}")

        # Convert to sunfish pieces
        sunfish_pieces = [piece_map.get(p, p) for p in pieces]

        # Add rank line with padding
        rank_line = ' ' + ''.join(sunfish_pieces) + ' \n'
        board.extend(rank_line)

    board.extend('         \n' * 2)  # Bottom padding
    board_str = ''.join(board)

    # Return position with default values (can't infer castling/en passant from visual)
    return Position(board_str, 0, (True, True), (True, True), 0, 0)


# Global variable to track current position
current_position = None

def move_tool(position_str, move_str, color):
    """Apply a move to a position and return the new position."""
    global current_position
    try:
        position = string_to_position(position_str)
        new_position = apply_move(position, move_str, color)
        # Update the global position
        current_position = new_position
        return position_to_string(new_position)
    except Exception as e:
        return f"Error applying move: {str(e)}"


def best_tool(position_str):
    """Calculate the best move for the current position."""
    try:
        position = string_to_position(position_str)
        searcher = Searcher()

        # Get the best move by running the searcher
        for depth, gamma, score, move in searcher.search([position]):
            if move:
                # Convert move back to algebraic notation
                # This is a simplified conversion - in practice we'd need proper move rendering
                i, j = move.i, move.j
                # Convert to algebraic notation (simplified)
                from_file = chr(ord('a') + (i % 10 - 1))
                from_rank = str(8 - (i // 10 - 2))
                to_file = chr(ord('a') + (j % 10 - 1))
                to_rank = str(8 - (j // 10 - 2))
                move_str = f"{from_file}{from_rank}{to_file}{to_rank}"
                return move_str
            if depth >= 3:  # Stop after reasonable depth
                break

        return "No move found"
    except Exception as e:
        return f"Error calculating best move: {str(e)}"


def parse_tool_calls(response):
    """Parse tool calls from LLM response.

    Returns a list of tool call dictionaries with 'name' and 'arguments' keys.
    """
    tool_calls = []

    # Look for tool_call tags
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(tool_call_pattern, response, re.DOTALL)

    for match in matches:
        try:
            tool_call = json.loads(match.strip())
            if 'name' in tool_call and 'arguments' in tool_call:
                tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue

    # Also look for direct JSON objects (fallback)
    if not tool_calls:
        # Find JSON-like objects in the response
        json_pattern = r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}'
        json_matches = re.findall(json_pattern, response, re.DOTALL)

        for match in json_matches:
            try:
                tool_call = json.loads(match)
                if 'name' in tool_call and 'arguments' in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

    return tool_calls


def execute_tool_call(tool_call):
    """Execute a tool call and return the result."""
    name = tool_call['name']
    args = tool_call['arguments']

    if name == 'move':
        position = args.get('position')
        move = args.get('move')
        color = args.get('color')
        if position and move and color:
            return move_tool(position, move, color)
        else:
            return "Error: Missing required arguments for move tool"

    elif name == 'best':
        position = args.get('position')
        if position:
            return best_tool(position)
        else:
            return "Error: Missing required arguments for best tool"

    else:
        return f"Error: Unknown tool '{name}'"


def main():
    global current_position

    llm = get_llm(MODEL)
    llm.load_model()
    while llm.is_loading() or not llm.is_running():
        sleep(1)

    # Initialize game state
    current_position = Position(initial, 0, (True, True), (True, True), 0, 0)
    # llm_color = 'white' if randint(0, 1) == 0 else 'black'
    llm_color = 'black'
    opponent_color = 'black' if llm_color == 'white' else 'white'

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Returns the new position after the move is played",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "position": {"type": "string", "description": "Current board position"},
                        "move": {"type": "string", "description": "Move in algebraic notation"},
                        "color": {"type": "string", "enum": ["white", "black"]}
                    },
                    "required": ["position", "move", "color"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "best",
                "description": "Calculates the best move for you. Only takes your drawback into account",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "position": {"type": "string", "description": "Current board position"}
                    },
                    "required": ["position"]
                }
            }
        }
    ]

    # Initialize conversation
    # drawback = DRAWBACKS[randint(0, len(DRAWBACKS) - 1)]
    drawback = '**Lucky Bastard**: You do not have a drawback.'
    system_prompt = PROMPT.replace('{drawback}', drawback).replace('{color}', llm_color).replace('{position}', position_to_string(current_position))
    conversation = [{'role': 'system', 'content': system_prompt}]

    print(f"Welcome to Drawback Chess!")
    print(f"You are playing as {opponent_color}")
    print(f"AI is playing as {llm_color} with drawback: {drawback}")
    print("\nStarting position:")
    print(position_to_string(current_position))
    print("\nYou can chat naturally. The AI will handle all moves using tools.")
    print("Type 'quit' to exit.\n")

    loop = True
    while loop:
        loop = False
        # Get user input (natural language)
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break

        # Add user message to conversation
        conversation.append({'role': 'user', 'content': user_input})

        # Get LLM response with tools
        response = llm.generate(conversation, chat_template=CHAT_TEMPLATE, template_env={'tools': [tool['function'] for tool in tools]}, stop=['</tool_call>'])

        # Handle tool calls
        tool_calls = parse_tool_calls(response)

        if tool_calls:
            # Add the assistant's message with tool calls to conversation
            conversation.append({'role': 'assistant', 'content': response})

            # Execute each tool call and add results to conversation
            for tool_call in tool_calls:
                tool_result = execute_tool_call(tool_call)
                print(f"Tool call: {tool_call['name']} -> {tool_result}")

                # Add tool result to conversation
                conversation.append({
                    'role': 'tool',
                    'content': tool_result,
                    'tool_call_id': f"{tool_call['name']}_{len(conversation)}"  # Simple ID generation
                })

            # Get final response after tool execution
            final_response = llm.generate(conversation, chat_template=CHAT_TEMPLATE, template_env={'tools': [tool['function'] for tool in tools]})
            print(f"AI: {final_response}")

            # Add final response to conversation
            conversation.append({'role': 'assistant', 'content': final_response})
        else:
            # No tool calls, just print the response
            print(f"AI: {response}")
            conversation.append({'role': 'assistant', 'content': response})
    llm.stop()


if __name__ == '__main__':
    main()
