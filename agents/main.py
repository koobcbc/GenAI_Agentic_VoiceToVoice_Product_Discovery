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


    answer = asyncio.run(
        app.ainvoke({"input": transcription})
    )
    response = answer['response']
    retrieved_context = answer['retrieved_context']
    print("================ FINAL ANSWER ===============")
    print(response)
    print("================ RETRIEVED CONTEXT ===============")
    print(retrieved_context)


    # test = "Today is a wonderful day to build something people love!"
    asyncio.run(run_tts(response))



if __name__ == "__main__":
    main()