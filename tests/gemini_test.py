from google import genai
from google.genai.types import Content, Part
from dotenv import load_dotenv

import os
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  # 或 GOOGLE_API_KEY
client = genai.Client(api_key=api_key)


resp = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents=Content(
        role="user",
        parts=[Part.from_text(text="Say hello in one short sentence.")]
    ),
)

print(resp.text)