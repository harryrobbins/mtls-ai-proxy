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


def get_openai_mtls_client(asynchronous=False):
    ctx = ssl.create_default_context(cafile=CA_PATH)
    ctx.load_cert_chain(certfile=CERTIFICATE_PATH, keyfile=KEY_PATH)

    if asynchronous:
        httpx_client = httpx.AsyncClient(verify=ctx)
        openai_client = openai.AsyncOpenAI(http_client=httpx_client,
                                           base_url="https://localhost:8443",
                                           api_key=LITELLM_PROXY_KEY)
    else:
        httpx_client = httpx.Client(verify=ctx)
        openai_client = openai.OpenAI(http_client=httpx_client,
                                      base_url="https://localhost:8443",
                                      api_key=LITELLM_PROXY_KEY)
    return openai_client


# client = get_openai_mtls_client()
#
# response = client.chat.completions.create(
#     model='ollama-qwen-local',
#     messages=[
#         {
#             "role": "user",
#             "content": "Write a one-sentence bedtime story about a unicorn."
#         },
#     ],
#     max_tokens=5
# )
#
# print(response.choices[0].message.content)
# print("=" * 40)

from litellm_utils import convert_openai_chat_completion_to_litellm_model_response


class MTLSOpenAILLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        client = get_openai_mtls_client()
        return convert_openai_chat_completion_to_litellm_model_response(client.chat.completions.create(
            model='ollama-qwen-local',
            messages=[
                {
                    "role": "user",
                    "content": "don't think say 'litellm openai'."
                },
            ],
            max_tokens=15
        ))

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        client = get_openai_mtls_client(asynchronous=True)
        return convert_openai_chat_completion_to_litellm_model_response(client.chat.completions.create(
            model='ollama-qwen-local',
            messages=[
                {
                    "role": "user",
                    "content": "don't think say 'litellm openai'."
                },
            ],
            max_tokens=5
        ))


my_mtls_openai_llm = MTLSOpenAILLM()

litellm.custom_provider_map = [  # 👈 KEY STEP - REGISTER HANDLER
    {"provider": "mtls_openai_llm", "custom_handler": my_mtls_openai_llm}
]

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
print("=" * 40)

