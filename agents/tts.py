import asyncio
from openai.helpers import LocalAudioPlayer
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

openai = AsyncOpenAI()

async def run_tts(input:str) -> None:
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=input,
        instructions="Speak in a cheerful and positive tone.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

# if __name__ == "__main__":
#     asyncio.run(main())


# speech_file_path = Path("speech.mp3")

# with client.audio.speech.with_streaming_response.create(
#     model="gpt-4o-mini-tts",
#     voice="coral",
#     input="Today is a wonderful day to build something people love!",
# ) as response:
#     response.stream_to_file(speech_file_path)