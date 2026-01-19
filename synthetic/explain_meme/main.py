import sys
import pathlib
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from database import query_db
from llm import get_llm
from dotenv import load_dotenv


load_dotenv(project_root / '.env')


LANGUAGES = {
    'de': 'Erkläre mir dieses Meme.',
    'en': 'Explain me this meme.',
}
MODEL = 'OpenRouter/openai/gpt-5-nano'
URL_PATTERN = 'https://ksalp.ch/download/memes_19847197567812934/%s'

def main():
    # Read meme filenames
    with open('memes.txt', 'r') as f:
        meme_files = [line.strip() for line in f.readlines() if line.strip()]

    # Initialize LLM
    llm = get_llm(MODEL)

    # Process each meme for English only
    for meme_file in meme_files:
        # Check if already processed for English
        existing = query_db("SELECT 1 FROM responses WHERE meme = ? AND language = ?", (meme_file, 'en'), one=True)
        if existing:
            print(f"Skipping {meme_file} - already processed for English")
            continue

        # Construct image URL
        image_url = URL_PATTERN % meme_file

        # Create vision message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LANGUAGES['en']},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        try:
            # Generate explanation
            explanation = llm.generate(messages)
            print(f"File: {meme_file}")
            print(f"Explanation: {explanation}")
            print("-" * 50)

            # Store in database
            query_db("INSERT INTO responses (meme, language, response) VALUES (?, ?, ?)",
                    (meme_file, 'en', explanation))

        except Exception as e:
            print(f"Error processing {meme_file}: {e}")
            continue

if __name__ == '__main__':
    main()
