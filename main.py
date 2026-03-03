# To run this code you need to install the following dependencies:
# pip install google-genai

tools = [
    types.Tool(googleSearch=types.GoogleSearch()),
]

import os
from google import genai
from google.genai import types

def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-3-flash-preview"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""I am scared about my surgery tomorrow. The doctor said there is a 4% risk."""),
            ],
        ),
    ]
    tools = [
        types.Tool(googleSearch=types.GoogleSearch(
        )),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.5,
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH",
        ),
        tools=tools,
        system_instruction=[
            types.Part.from_text(text="""CRITICAL DIRECTIVE: You are a Medical Empathy Specialist. When a user expresses fear about surgery:

MANDATORY: Start by saying \"It is completely normal to feel this way.\"

MANDATORY: Explicitly mention that surgeons study for over a decade and have handled much more dire cases.

LOGIC: Reframe any percentage (like 4% risk) as a \"96% success rate,\" which is overwhelmingly positive.

CALL TO ACTION: Tell them to trust the doctor's \"capable hands\" and focus on their 1-month rest period."""),
        ],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()

# ENABLE GROUNDING: This tells Gemini to verify medical claims 
# using live Google Search data before answering.
tools = [
    types.Tool(googleSearch=types.GoogleSearch()),
]
