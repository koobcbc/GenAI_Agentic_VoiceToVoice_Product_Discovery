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

    transcription = "I want to find a costume for this halloween. Could you recommend for age 8-14?"

    answer = asyncio.run(
        app.ainvoke({"input": transcription})
    )
    response = answer['response']
    retrieved_context = answer['retrieved_context']
    print("================ FINAL ANSWER ===============")
    print(response)
    print("================ RETRIEVED CONTEXT ===============")
    print(retrieved_context)

    # response = '* Final Answer:\nFor a Halloween costume suitable for ages 8-14, consider the Rubies Teen Titans Go Robin Costume, which is available in child medium size. It is highly rated with a 4.9 rating and is priced at $15.99. Alternatively, the Forum Novelties Zombie Girl Costume in child medium is another option, priced at $27.07 with a 3.3 rating. Lastly, the Slender Man Kids Morphsuit Costume in size large (4\' - 4\'6") is available for $29.97, though it has a lower rating of 3.1.\n\n* Cited Sources:\n  - "Rubies Teen Titans Go Robin Costume, Child Medium", price: 15.99, rating: 4.9 [source: rag.search | doc_id: adb84209e1c97c5d06e6970c13805ef8]\n  - "Forum Novelties Zombie Girl Costume, Child\'s Medium", price: 27.07, rating: 3.3 [source: rag.search | doc_id: 5da84c8c20195a2f3f0261debce8ef2a]\n  - "Slender Man Kids Morphsuit Costume - size Large 4\' - 4\'6\\"", price: 29.97, rating: 3.1 [source: rag.search | doc_id: 4ded7dc6e50b7e09c7503bef0890480a]\n\n* Safety Flags: No'
    # test = "Today is a wonderful day to build something people love!"
    asyncio.run(run_tts(response))



if __name__ == "__main__":
    main()