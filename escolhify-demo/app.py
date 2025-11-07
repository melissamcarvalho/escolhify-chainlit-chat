import chainlit as cl
from openai import AsyncOpenAI

from dotenv import load_dotenv

from escolhify.src.core import EscolhifyConfig
from escolhify.src.core import EscolhifyLogger
from escolhify.src.core.llm_interface import LLMFactory

# Load environment variables
load_dotenv()

# Initialize configuration and logger
config = EscolhifyConfig.load()
logger = EscolhifyLogger(level=config.log_level)

client = AsyncOpenAI()

# Instrument the OpenAI client
cl.instrument_openai()

settings = {
    "model": config.llm.model,
    "temperature": config.llm.temperature,
}

@cl.on_message
async def on_message(message: cl.Message):
    response = await client.chat.completions.create(
        messages=[
            {
                "content": "Você é um assistente especializado em ajudar os usuários a escolher produtos com base em suas necessidades e preferências. "
                "Comece sempre se apresentando como Escolhify, um assistente de escolha de produtos. ",
                "role": "system"
            },
            {
                "content": message.content,
                "role": "user"
            }
        ],
        **settings
    )
    await cl.Message(
        content=response.choices[0].message.content,
        author="Escolhify"
    ).send()


if __name__ == "__main__":

    cl.run()
    