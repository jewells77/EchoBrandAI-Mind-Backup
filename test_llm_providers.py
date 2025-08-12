import asyncio
from dotenv import load_dotenv

from app.domain.llm_providers.factory import LLMProviderFactory, LLMProviderType


async def test_openai_provider():
    print("\n=== Testing OpenAI Provider ===")
    provider = LLMProviderFactory.get_provider()  # Default is OpenAI

    # Test basic generation
    print("Testing generation...")
    response = await provider.generate(
        [
            {
                "role": "user",
                "content": "What are three key practices for sustainable fashion?",
            }
        ]
    )
    print(f"Response: {response.content}")

    # Test streaming
    print("\nTesting streaming...")
    print("Response: ", end="", flush=True)
    async for chunk in provider.stream(
        [{"role": "user", "content": "List two eco-friendly fabric alternatives."}]
    ):
        print(chunk.content, end="", flush=True)
    print("\n")


async def main():
    # Load environment variables from .env file
    load_dotenv()

    # Test OpenAI provider
    try:
        await test_openai_provider()
    except Exception as e:
        print(f"Error testing OpenAI provider: {e}")


if __name__ == "__main__":
    asyncio.run(main())
