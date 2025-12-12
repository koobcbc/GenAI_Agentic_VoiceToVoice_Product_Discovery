import asyncio
from openai.helpers import LocalAudioPlayer
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()

openai = AsyncOpenAI()
client = OpenAI()

def summarize_text(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the text and focus only on important information the user should hear."},
            {"role": "user", "content": text},
        ]
    )

    return response.choices[0].message.content


async def run_tts(input:str) -> None:
    summary = summarize_text(input)

    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=summary,
        instructions="Summarize given input and Speak in a cheerful and positive tone.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)