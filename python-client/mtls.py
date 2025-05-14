from pathlib import Path

import httpx
import os

from litellm import api_key

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

ctx = ssl.create_default_context(cafile=CA_PATH)
ctx.load_cert_chain(certfile=CERTIFICATE_PATH, keyfile=KEY_PATH)

client = httpx.Client(verify=ctx)

import openai

openai_client = openai.OpenAI(http_client=client, base_url="https://localhost:8443", api_key=LITELLM_PROXY_KEY)

completion = openai_client.chat.completions.create(
    model='ollama-qwen-local',
    messages=[
        {
            "role": "user",
            "content": "Write a one-sentence bedtime story about a unicorn."
        }
    ]
)

print(completion.choices[0].message.content)
