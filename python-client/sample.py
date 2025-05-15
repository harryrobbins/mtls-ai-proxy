import asyncio
from pathlib import Path

import httpx
import os

import litellm
from litellm import api_key, CustomLLM, completion, acompletion
from litellm.types.utils import ModelResponse
from openai.types.chat import ChatCompletion
# from OpenSSL import SSL, X509

current_file_dir = Path(__file__).parent
certs_dir = current_file_dir.parent / "certs"
CERTIFICATE_PATH = certs_dir / "client.crt"
KEY_PATH = certs_dir / "client.key"
CA_PATH = certs_dir / "ca.crt"

for path in (CERTIFICATE_PATH, KEY_PATH, CA_PATH):
    if not path.exists():
        exit("Could not find certificate file {}".format(path))

from dotenv import load_dotenv

load_dotenv()

base_model = os.getenv("BASE_MODEL")
ollama_model = "ollama/" + base_model
LITELLM_PROXY_MODEL = "litellm_proxy/ollama-qwen-local"
LITELLM_PROXY_KEY = "sk-1234"

import ssl
import httpx
import openai


class MTLSOpenAILLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        pass

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        pass
    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        pass

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        """
        calls completion from upstream llm but returns the response as if it was a stream
        """
        pass



    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        """
        calls acompletion from upstream llm but returns the response as if it was a stream
        """

        pass


my_mtls_openai_llm = MTLSOpenAILLM()

litellm.custom_provider_map = [  # 👈 KEY STEP - REGISTER HANDLER
    {"provider": "mtls_openai_llm", "custom_handler": my_mtls_openai_llm}
]

if __name__ == "__main__":

# test the proxy works with synchronous calls
    resp = completion(
        model="mtls_openai_llm/anything-you-like-here",
        messages=[{"role": "user", "content": "Hello world!"}],
    )

    print(resp.choices[0].message.content)
    print("=" * 40)

    resp = asyncio.run(
        acompletion(
        model="mtls_openai_llm/anything-you-like-here",
                                   messages=[{"role": "user", "content": "Hello world!"}], ))

    print(resp.choices[0].message.content)
    print("=" * 40)

    """
    Streaming demo
    """
    resp = completion(
            model="mtls_openai_llm/anything-you-like-here",
            messages=[{"role": "user", "content": "Hello world!"}],
            streaming=True,
            )
    print(resp.choices[0].message.content)
    print("=" * 40)

    resp = asyncio.run(
        acompletion(
            model="mtls_openai_llm/anything-you-like-here",
            messages=[{"role": "user", "content": "Hello world!"}],
            streaming=True,
            )
    )
    print(resp.choices[0].message.content)
    print("=" * 40)

