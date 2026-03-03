# To run this code, install dependencies: pip install google-genai
import os
from google import genai
from google.genai import types

def generate():
    # This pulls your API key from your system settings for safety
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Selecting the latest preview model as shown in your AI Studio settings
    model = "gemini-3-flash-preview"
    
    # ENABLE GROUNDING: This tells Gemini to verify facts using live Google Search
    tools = [types.Tool(googleSearch=types.GoogleSearch())]

    # This configuration contains your empathy logic and model settings
    generate_content_config = types.GenerateContentConfig(
        temperature=0.5,
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        tools=tools,
        system_instruction="""CRITICAL DIRECTIVE: You are a Medical Empathy Specialist. 
        MANDATORY: Start by saying 'It is completely normal to feel this way.' 
        MANDATORY: Explicitly mention that surgeons study for over a decade. 
        LOGIC: Reframe any percentage (like 4% risk) as a '96% success rate'. 
        CALL TO ACTION: Tell them to focus on their 1-month rest period."""
    )

    # This is the specific question the AI will answer when you run the script
    contents = [
        types.Content(
            role="user", 
            parts=[types.Part.from_text("I am scared about my surgery tomorrow. The doctor said there is a 4% risk.")]
        )
    ]

    # This loop prints the answer to your screen
    for chunk in client.models.generate_content_stream(
        model=model, 
        contents=contents, 
        config=generate_content_config
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
