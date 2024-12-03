from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

print(api_key)
