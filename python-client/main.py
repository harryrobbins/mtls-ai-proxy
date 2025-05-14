import os

import litellm
from litellm import CustomLLM, completion, get_llm_provider, acompletion
import asyncio

base_model = os.getenv("BASE_MODEL")
ollama_model = "ollama/" + base_model
litellm_proxy_model = "litellm_proxy/ollama-qwen-local"
litellm_proxy_key = "sk-1234"


print("="*40)
# test we can use the sdk to communicate with ollama directly
resp = completion(
        model=ollama_model,
        messages=[{"role": "user", "content": "Hello world!"}],
        max_tokens=5,
        base_url="http://localhost:11434",
    )

print( resp.choices[0].message.content )
print("="*40)

# test we can communicate with ollama via the litellm proxy
resp = completion(
        model=litellm_proxy_model,
        messages=[{"role": "user", "content": "Hello world!"}],
        max_tokens=5,
        base_url="http://localhost:4000",
        api_key=litellm_proxy_key
    )

print( resp.choices[0].message.content )
print("="*40)

### tests using pass-through custom handler

class OllamaProxyLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model=ollama_model,
            messages=[{"role": "user", "content": "say 'i am synchronous'"}],
            max_tokens=50
        )  # type: ignore

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model=ollama_model,
            messages=[{"role": "user", "content": "say 'i am synchronous'"}],
            max_tokens=50
        )  # type: ignore


my_ollama_proxy_llm = OllamaProxyLLM()

litellm.custom_provider_map = [  # 👈 KEY STEP - REGISTER HANDLER
    {"provider": "ollama_proxy_llm", "custom_handler": my_ollama_proxy_llm}
]


# test the proxy works with synchronous calls
resp = completion(
    model="ollama_proxy_llm/my-fake-model",
    messages=[{"role": "user", "content": "Hello world!"}],
)

print(resp.choices[0].message.content)
print("="*40)

# test the proxy works with asynchronous calls
resp = asyncio.run(acompletion(
    model="ollama_proxy_llm/anything-goes-here",
    messages=[{"role": "user", "content": "Hello world!"}],
))

print(resp.choices[0].message.content)
print("="*40)
