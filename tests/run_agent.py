import os
import sys
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
from dotenv import load_dotenv
from src.agent.agent import ReActAgent
from src.core.local_provider import LocalProvider
from src.core.openai_provider import OpenAIProvider
from src.core.gemini_provider import GeminiProvider
from src.core.local_provider import LocalProvider

load_dotenv()
print("LOCAL_MODEL_PATH =", os.getenv("LOCAL_MODEL_PATH"))

PROVIDER_TYPE = "gemini" # Change to "local", "openai", or "gemini" as needed, local is default for testing

if PROVIDER_TYPE == "local":
    provider = LocalProvider(
        model_path="./models/Phi-3-mini-4k-instruct-q4.gguf"
    )

elif PROVIDER_TYPE == "openai":
    provider = OpenAIProvider(
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY") #API key should be set in .env file
    )

elif PROVIDER_TYPE == "gemini":
    provider = GeminiProvider(
        model_name="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY") #API key should be set in .env file
    )

else:
    raise ValueError(f"Unknown provider: {PROVIDER_TYPE}")

agent = ReActAgent(
    llm=provider,
    tools=ReActAgent.my_tools
)

response = agent.run(
    "What's the weather in London?" # Change the input to test different queries
)

print(response)