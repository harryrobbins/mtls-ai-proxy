import asyncio
import ssl
import time
import uuid
import os
import traceback
from pathlib import Path
from typing import AsyncIterator, Iterator

import httpx
import openai  # For OpenAI client and types
from openai.types.chat import ChatCompletion, ChatCompletionMessage  # For mocking/typing upstream response

import litellm
# Corrected imports for LiteLLM types
from litellm import CustomLLM
from litellm.utils import ModelResponse, Choices, Message, Usage  # LiteLLM specific types
from litellm.types.utils import Delta, StreamingChoices, GenericStreamingChunk  # Import GenericStreamingChunk

from dotenv import load_dotenv

# Assuming litellm_utils.py is in the same directory
try:
    from litellm_utils import convert_openai_chat_completion_to_litellm_model_response
except ImportError:
    print("Error: litellm_utils.py not found. Please ensure it's in the same directory.")
    print("It should contain the function: convert_openai_chat_completion_to_litellm_model_response")
    exit(1)

# --- Certificate Paths ---
current_file_dir = Path(__file__).parent
certs_dir = current_file_dir.parent / "certs"
CERTIFICATE_PATH = certs_dir / "client.crt"
KEY_PATH = certs_dir / "client.key"
CA_PATH = certs_dir / "ca.crt"

# --- Validate Certificate Paths ---
for path_to_check in (CERTIFICATE_PATH, KEY_PATH, CA_PATH):
    if not path_to_check.exists():
        print(f"Error: Could not find certificate file {path_to_check}")
        print(f"Current file directory: {current_file_dir}")
        print(f"Calculated certs directory: {certs_dir}")
        exit(1)
    else:
        print(f"Found certificate file: {path_to_check}")

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
UPSTREAM_MODEL_NAME = "ollama-qwen-local"
LITELLM_PROXY_KEY = "sk-1234"
MTLS_PROXY_URL = "https://localhost:8443"


class MTLSOpenAILLM(CustomLLM):
    """
    Custom LiteLLM handler that:
    1. Uses mTLS to communicate with an upstream OpenAI-compatible API (e.g., LiteLLM Proxy).
    2. Implements "fake streaming" by making a non-streaming call and returning the response
       as a single content chunk followed by a finalization chunk, using GenericStreamingChunk.
    """

    def _get_client(self, asynchronous: bool = False):
        try:
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(CA_PATH))
            ctx.load_cert_chain(certfile=str(CERTIFICATE_PATH), keyfile=str(KEY_PATH))
        except Exception as e:
            raise RuntimeError(f"Failed to create SSL context: {e}")

        timeout_config = httpx.Timeout(600.0, connect=10.0)

        if asynchronous:
            httpx_client = httpx.AsyncClient(verify=ctx, timeout=timeout_config)
            return openai.AsyncOpenAI(
                http_client=httpx_client,
                base_url=MTLS_PROXY_URL,
                api_key=LITELLM_PROXY_KEY
            )
        else:
            httpx_client = httpx.Client(verify=ctx, timeout=timeout_config)
            return openai.OpenAI(
                http_client=httpx_client,
                base_url=MTLS_PROXY_URL,
                api_key=LITELLM_PROXY_KEY
            )

    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        messages = kwargs.get("messages")
        # Make a copy of kwargs to modify, ensuring 'stream' is not passed to the non-streaming call
        local_kwargs = kwargs.copy()
        local_kwargs.pop("stream", None)  # Explicitly remove stream for this non-streaming method

        sync_openai_client = self._get_client(asynchronous=False)
        try:
            # Use local_kwargs which doesn't have 'stream'
            upstream_response: ChatCompletion = sync_openai_client.chat.completions.create(
                model=UPSTREAM_MODEL_NAME,
                messages=messages,
                max_tokens=local_kwargs.get("max_tokens", 150),
                temperature=local_kwargs.get("temperature", 0.7),
            )
        except httpx.ReadTimeout as e:
            raise litellm.Timeout(message=f"MTLS Upstream ReadTimeout: {str(e)}", model=UPSTREAM_MODEL_NAME,
                                  llm_provider="custom_mtls_provider")
        except Exception as e:
            raise litellm.APIConnectionError(f"MTLS Upstream Error: {str(e)}", model=UPSTREAM_MODEL_NAME,
                                             llm_provider="custom_mtls_provider")

        litellm_response = convert_openai_chat_completion_to_litellm_model_response(
            openai_response=upstream_response
        )
        if not hasattr(litellm_response, 'usage') or litellm_response.usage is None:
            # Ensure usage is present for the streaming methods to access
            prompt_tokens = litellm.token_counter(model=UPSTREAM_MODEL_NAME, messages=messages)
            completion_tokens = litellm.token_counter(model=UPSTREAM_MODEL_NAME,
                                                      text=litellm_response.choices[0].message.content or "")
            litellm_response.usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                           total_tokens=prompt_tokens + completion_tokens)

        return litellm_response

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        messages = kwargs.get("messages")
        # Make a copy of kwargs to modify, ensuring 'stream' is not passed to the non-streaming call
        local_kwargs = kwargs.copy()
        local_kwargs.pop("stream", None)  # Explicitly remove stream for this non-streaming method

        async_openai_client = self._get_client(asynchronous=True)
        try:
            # Use local_kwargs which doesn't have 'stream'
            upstream_response: ChatCompletion = await async_openai_client.chat.completions.create(
                model=UPSTREAM_MODEL_NAME,
                messages=messages,
                max_tokens=local_kwargs.get("max_tokens", 150),
                temperature=local_kwargs.get("temperature", 0.7),
            )
        except httpx.ReadTimeout as e:
            raise litellm.Timeout(message=f"MTLS Upstream ReadTimeout: {str(e)}", model=UPSTREAM_MODEL_NAME,
                                  llm_provider="custom_mtls_provider")
        except Exception as e:
            raise litellm.APIConnectionError(f"MTLS Upstream Error: {str(e)}", model=UPSTREAM_MODEL_NAME,
                                             llm_provider="custom_mtls_provider")

        litellm_response = convert_openai_chat_completion_to_litellm_model_response(
            openai_response=upstream_response
        )
        if not hasattr(litellm_response, 'usage') or litellm_response.usage is None:
            # Ensure usage is present for the streaming methods to access
            prompt_tokens = litellm.token_counter(model=UPSTREAM_MODEL_NAME, messages=messages)
            completion_tokens = litellm.token_counter(model=UPSTREAM_MODEL_NAME,
                                                      text=litellm_response.choices[0].message.content or "")
            litellm_response.usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                           total_tokens=prompt_tokens + completion_tokens)

        return litellm_response

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        # This method is called when litellm.completion(..., stream=True) is used.
        # It should call the non-streaming version and then adapt its output.
        model_response: ModelResponse = self.completion(*args, **kwargs)

        response_content = ""
        if model_response.choices and model_response.choices[0].message and model_response.choices[0].message.content:
            response_content = model_response.choices[0].message.content

        _model_name = model_response.model or kwargs.get("model", "mtls_custom_model")
        _id = model_response.id or f"chatcmpl-fakestream-sync-{uuid.uuid4()}"
        _created = model_response.created or int(time.time())
        _system_fingerprint = getattr(model_response, 'system_fingerprint', None)

        # Yield content chunk
        yield GenericStreamingChunk(
            id=_id,
            object="chat.completion.chunk",
            created=_created,
            model=_model_name,
            system_fingerprint=_system_fingerprint,
            text=response_content,
            is_finished=False,
            finish_reason=None,
            usage=None,
            index=0
        )

        # Yield finalization chunk
        yield GenericStreamingChunk(
            id=_id,
            object="chat.completion.chunk",
            created=_created,
            model=_model_name,
            system_fingerprint=_system_fingerprint,
            text="",
            is_finished=True,
            finish_reason="stop",
            usage=model_response.usage,
            index=0
        )

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        # This method is called when litellm.acompletion(..., stream=True) is used.
        model_response: ModelResponse = await self.acompletion(*args, **kwargs)
        response_content = ""
        if model_response.choices and model_response.choices[0].message and model_response.choices[0].message.content:
            response_content = model_response.choices[0].message.content

        _model_name = model_response.model or kwargs.get("model", "mtls_custom_model")
        _id = model_response.id or f"chatcmpl-fakestream-async-{uuid.uuid4()}"
        _created = model_response.created or int(time.time())
        _system_fingerprint = getattr(model_response, 'system_fingerprint', None)

        # Yield content chunk
        yield GenericStreamingChunk(
            id=_id,
            object="chat.completion.chunk",
            created=_created,
            model=_model_name,
            system_fingerprint=_system_fingerprint,
            text=response_content,
            is_finished=False,
            finish_reason=None,
            usage=None,
            index=0
        )

        # Yield finalization chunk
        yield GenericStreamingChunk(
            id=_id,
            object="chat.completion.chunk",
            created=_created,
            model=_model_name,
            system_fingerprint=_system_fingerprint,
            text="",
            is_finished=True,
            finish_reason="stop",
            usage=model_response.usage,
            index=0
        )


# --- Register the custom LLM provider with LiteLLM ---
my_mtls_openai_llm = MTLSOpenAILLM()

litellm.custom_provider_map = [
    {"provider": "mtls_openai_llm", "custom_handler": my_mtls_openai_llm}
]

# --- Main execution block for testing ---
if __name__ == "__main__":
    # litellm.set_verbose = True

    print("Testing synchronous non-streaming:")
    try:
        resp_sync_non_stream = litellm.completion(
            model="mtls_openai_llm/sync-non-stream",
            messages=[{"role": "user", "content": "Hello from sync non-stream!"}],
        )
        print(f"Response: {resp_sync_non_stream.choices[0].message.content}")
        print(f"Usage: {resp_sync_non_stream.usage}")
    except Exception as e:
        print(f"Error in sync non-streaming: {e}")
        traceback.print_exc()
    print("=" * 40)

    print("Testing asynchronous non-streaming:")
    try:
        resp_async_non_stream = asyncio.run(
            litellm.acompletion(
                model="mtls_openai_llm/async-non-stream",
                messages=[{"role": "user", "content": "Hello from async non-stream!"}],
            )
        )
        print(f"Response: {resp_async_non_stream.choices[0].message.content}")
        print(f"Usage: {resp_async_non_stream.usage}")
    except Exception as e:
        print(f"Error in async non-streaming: {e}")
        traceback.print_exc()
    print("=" * 40)

    print("Testing synchronous streaming:")
    try:
        response_sync_stream = litellm.completion(
            model="mtls_openai_llm/sync-stream",
            messages=[{"role": "user", "content": "Hello from sync stream!"}],
            stream=True,
        )
        full_response_content_sync = ""
        usage_sync = None
        print("Iterating over sync stream:")
        # The CustomStreamWrapper will convert GenericStreamingChunk to ModelResponseStream
        # So, the consuming code should expect ModelResponseStream's structure.
        for chunk_num, chunk in enumerate(response_sync_stream):
            content = chunk.choices[0].delta.content  # Correct way to access content
            finish_reason = chunk.choices[0].finish_reason
            print(f"  Sync Stream Chunk {chunk_num + 1}: finish_reason='{finish_reason}', content='{content or ''}'")
            if content:
                full_response_content_sync += content
            if chunk.usage:
                usage_sync = chunk.usage
        print(f"Full Sync Streamed Response: {full_response_content_sync}")
        if usage_sync:
            print(f"Sync Stream Usage: {usage_sync}")
    except Exception as e:
        print(f"Error in sync streaming: {e}")
        traceback.print_exc()
    print("=" * 40)

    print("Testing asynchronous streaming:")


    async def consume_async_stream():
        try:
            response_async_stream = await litellm.acompletion(
                model="mtls_openai_llm/async-stream",
                messages=[{"role": "user", "content": "Hello from async stream!"}],
                stream=True,
            )
            full_response_content_async = ""
            usage_async = None
            print("Iterating over async stream:")
            chunk_num = 0
            async for chunk in response_async_stream:
                chunk_num += 1
                content = chunk.choices[0].delta.content  # Correct way to access content
                finish_reason = chunk.choices[0].finish_reason
                print(f"  Async Stream Chunk {chunk_num}: finish_reason='{finish_reason}', content='{content or ''}'")
                if content:
                    full_response_content_async += content
                if chunk.usage:
                    usage_async = chunk.usage
            print(f"Full Async Streamed Response: {full_response_content_async}")
            if usage_async:
                print(f"Async Stream Usage: {usage_async}")
        except Exception as e:
            print(f"Error in async streaming: {e}")
            traceback.print_exc()


    asyncio.run(consume_async_stream())
    print("=" * 40)
