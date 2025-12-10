from whisper_ars import speedch_recognition
from langchain_core.messages import HumanMessage
from graph.graph import app
from tts import run_tts
import asyncio

def main():
    print("Starting Automatic Speech Recognition...")
    transcription = speedch_recognition()
    print("Final Transcription Output:")
    print(transcription)

    answer = app.invoke({"input": transcription})['response']
    print(answer)

    # test = "Today is a wonderful day to build something people love!"
    asyncio.run(run_tts(answer))



if __name__ == "__main__":
    main()