import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


async def test_openai():
    # Load environment variables
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return

    print(f"Using API key: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}")

    # Create ChatOpenAI instance
    chat = ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL_NAME"), temperature=0.7, api_key=api_key
    )

    # Test generation
    try:
        print("Testing OpenAI...")
        response = await chat.ainvoke([HumanMessage(content="Hello, how are you?")])
        print(f"Response: {response.content}")
        print("Test successful!")
    except Exception as e:
        print(f"Error testing OpenAI: {e}")


if __name__ == "__main__":
    asyncio.run(test_openai())
