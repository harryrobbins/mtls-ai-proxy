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


class MTLSOpenAILLM(CustomLLM):
    """
    Custom LiteLLM handler that:
    1. Uses mTLS to communicate with an upstream OpenAI-compatible API (e.g., LiteLLM Proxy).
    2. Implements "fake streaming" by making a non-streaming call and returning the response
       as a single content chunk followed by a finalization chunk, using GenericStreamingChunk.
    """
    client_cert_path: str
    client_key_path: str
    ca_cert_path: str

    def __init__(self, client_cert_path,
                 client_key_path,
                 ca_cert_path, **kwargs):
        self.client_cert_path = client_cert_path
        self.client_key_path = client_key_path
        self.ca_cert_path = ca_cert_path
        # --- Validate Certificate Paths ---
        for path_to_check in (self.client_cert_path, self.client_key_path, self.ca_cert_path):
            if not Path(path_to_check).exists():
                print(f"Error: Could not find certificate file {path_to_check}")
                print(f"Current file directory: {current_file_dir}")
                print(f"Calculated certs directory: {certs_dir}")
                exit(1)
            else:
                print(f"Found cryptography file: {path_to_check}")
        super().__init__()


    def _get_client(self, asynchronous: bool = False):
        try:
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(self.ca_cert_path))
            ctx.load_cert_chain(certfile=str(self.client_cert_path), keyfile=str(self.client_key_path))
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
        local_kwargs = kwargs.copy()
        local_kwargs.pop("stream", None)
        upstream_model_name = local_kwargs.get("model")
        sync_openai_client = self._get_client(asynchronous=False)
        try:
            upstream_response: ChatCompletion = sync_openai_client.chat.completions.create(
                model=upstream_model_name,
                messages=messages,
                max_tokens=local_kwargs.get("max_tokens", 150),
                temperature=local_kwargs.get("temperature", 0.7),
            )
        except httpx.ReadTimeout as e:
            raise litellm.Timeout(message=f"MTLS Upstream ReadTimeout: {str(e)}", model=upstream_model_name,
                                  llm_provider="custom_mtls_provider")
        except Exception as e:
            raise litellm.APIConnectionError(f"MTLS Upstream Error: {str(e)}", model=upstream_model_name,
                                             llm_provider="custom_mtls_provider")

        litellm_response = convert_openai_chat_completion_to_litellm_model_response(
            openai_response=upstream_response
        )

        if not hasattr(litellm_response, 'usage') or litellm_response.usage is None:
            prompt_tokens = 0
            completion_tokens = 0
            if messages:
                try:  # Wrap token_counter in try-except as it can sometimes fail with custom/uncommon models
                    prompt_tokens = litellm.token_counter(model=upstream_model_name, messages=messages)
                except:
                    prompt_tokens = 0  # Fallback
            if litellm_response.choices and litellm_response.choices[0].message and litellm_response.choices[
                0].message.content:
                try:
                    completion_tokens = litellm.token_counter(model=upstream_model_name,
                                                              text=litellm_response.choices[0].message.content)
                except:
                    completion_tokens = 0  # Fallback
            litellm_response.usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                           total_tokens=prompt_tokens + completion_tokens)

        return litellm_response

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        messages = kwargs.get("messages")
        local_kwargs = kwargs.copy()
        local_kwargs.pop("stream", None)
        upstream_model_name = local_kwargs.get("model")

        async_openai_client = self._get_client(asynchronous=True)
        try:
            upstream_response: ChatCompletion = await async_openai_client.chat.completions.create(
                model=upstream_model_name,
                messages=messages,
                max_tokens=local_kwargs.get("max_tokens", 150),
                temperature=local_kwargs.get("temperature", 0.7),
            )
        except httpx.ReadTimeout as e:
            raise litellm.Timeout(message=f"MTLS Upstream ReadTimeout: {str(e)}", model=upstream_model_name,
                                  llm_provider="custom_mtls_provider")
        except Exception as e:
            raise litellm.APIConnectionError(f"MTLS Upstream Error: {str(e)}", model=upstream_model_name,
                                             llm_provider="custom_mtls_provider")

        litellm_response = convert_openai_chat_completion_to_litellm_model_response(
            openai_response=upstream_response
        )
        if not hasattr(litellm_response, 'usage') or litellm_response.usage is None:
            prompt_tokens = 0
            completion_tokens = 0
            if messages:
                try:
                    prompt_tokens = litellm.token_counter(model=upstream_model_name, messages=messages)
                except:
                    prompt_tokens = 0
            if litellm_response.choices and litellm_response.choices[0].message and litellm_response.choices[
                0].message.content:
                try:
                    completion_tokens = litellm.token_counter(model=upstream_model_name,
                                                              text=litellm_response.choices[0].message.content)
                except:
                    completion_tokens = 0
            litellm_response.usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                           total_tokens=prompt_tokens + completion_tokens)

        return litellm_response

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        model_response: ModelResponse = self.completion(*args, **kwargs)

        response_content = ""
        if model_response.choices and model_response.choices[0].message and model_response.choices[0].message.content:
            response_content = model_response.choices[0].message.content

        _model_name = model_response.model or kwargs.get("model", "mtls_custom_model")
        _id = model_response.id or f"chatcmpl-fakestream-sync-{uuid.uuid4()}"
        _created = model_response.created or int(time.time())
        _system_fingerprint = getattr(model_response, 'system_fingerprint', None)

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
        # add debugging
        # print(dict(model_response.usage))
        yield GenericStreamingChunk(
            id=_id,
            object="chat.completion.chunk",
            created=_created,
            model=_model_name,
            system_fingerprint=_system_fingerprint,
            text="",
            is_finished=True,
            finish_reason="stop",
            usage=dict(model_response.usage),
            index=0
        )

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        model_response: ModelResponse = await self.acompletion(*args, **kwargs)
        response_content = ""
        if model_response.choices and model_response.choices[0].message and model_response.choices[0].message.content:
            response_content = model_response.choices[0].message.content

        _model_name = model_response.model or kwargs.get("model", "mtls_custom_model")
        _id = model_response.id or f"chatcmpl-fakestream-async-{uuid.uuid4()}"
        _created = model_response.created or int(time.time())
        _system_fingerprint = getattr(model_response, 'system_fingerprint', None)

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
        # add debugging
        # print(model_response)
        yield GenericStreamingChunk(
            id=_id,
            object="chat.completion.chunk",
            created=_created,
            model=_model_name,
            system_fingerprint=_system_fingerprint,
            text="",
            is_finished=True,
            finish_reason="stop",
            usage=dict(model_response.usage),
            index=0
        )


# --- Main execution block for testing ---
if __name__ == "__main__":
    load_dotenv()

    # --- Certificate Paths ---
    current_file_dir = Path(__file__).parent
    certs_dir = current_file_dir.parent / "certs"
    CERTIFICATE_PATH = certs_dir / "client.crt"
    KEY_PATH = certs_dir / "client.key"
    CA_PATH = certs_dir / "ca.crt"

    # --- Load Environment Variables ---

    # --- Configuration ---
    LITELLM_PROXY_KEY = "sk-1234"
    MTLS_PROXY_URL = "https://localhost:8443"
    litellm.set_verbose = True
    MODEL_NAME_1 = "ollama-qwen2"
    MODEL_NAME_2 = "ollama-qwen3"

    # --- Register the custom LLM provider with LiteLLM ---
    my_mtls_openai_llm = MTLSOpenAILLM(client_cert_path=CERTIFICATE_PATH, client_key_path=KEY_PATH, ca_cert_path=CA_PATH)

    litellm.custom_provider_map = [
        {"provider": "mtls_openai_llm", "custom_handler": my_mtls_openai_llm}
    ]

    for model_name in [MODEL_NAME_1, MODEL_NAME_2]:
        print(f"Testing synchronous non-streaming with model {model_name}:")
        try:
            resp_sync_non_stream = litellm.completion(
                model=f"mtls_openai_llm/{model_name}",
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
                model=f"mtls_openai_llm/{MODEL_NAME_1}",
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
            model=f"mtls_openai_llm/{MODEL_NAME_1}",
            messages=[{"role": "user", "content": "Hello from sync stream!"}],
            stream=True, max_tokens=5, stream_options={"include_usage": True}
        )
        full_response_content_sync = ""
        usage_sync = None
        print("Iterating over sync stream:")
        for chunk_num, chunk in enumerate(response_sync_stream):
            content = None
            # For ModelResponseStream, content is in choices[0].delta.content
            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content

            finish_reason = chunk.choices[0].finish_reason
            print(f"  Sync Stream Chunk {chunk_num + 1}: finish_reason='{finish_reason}', content='{content or ''}'")
            if content:
                full_response_content_sync += content

            # Usage is typically on the last chunk or when finish_reason is not None
            if hasattr(chunk, 'usage') and chunk.usage is not None:
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
                model=f"mtls_openai_llm/{MODEL_NAME_1}",
                messages=[{"role": "user", "content": "Hello from async stream!"}],
                stream=True,
            )
            full_response_content_async = ""
            usage_async = None
            print("Iterating over async stream:")
            chunk_num = 0
            async for chunk in response_async_stream:
                chunk_num += 1
                content = None
                # For ModelResponseStream, content is in choices[0].delta.content
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content

                finish_reason = chunk.choices[0].finish_reason
                print(f"  Async Stream Chunk {chunk_num}: finish_reason='{finish_reason}', content='{content or ''}'")
                if content:
                    full_response_content_async += content

                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage_async = chunk.usage
            print(f"Full Async Streamed Response: {full_response_content_async}")
            if usage_async:
                print(f"Async Stream Usage: {usage_async}")
        except Exception as e:
            print(f"Error in async streaming: {e}")
            traceback.print_exc()


    asyncio.run(consume_async_stream())
    print("=" * 40)

########################################################################################

# base_model = os.getenv("BASE_MODEL")
# ollama_model = "ollama/" + base_model
#
# print("Testing synchronous non-streaming:")
# try:
#     resp_sync_non_stream = litellm.completion(
#         model=ollama_model,
#         messages=[{"role": "user", "content": "Hello from sync non-stream!"}],
#         max_tokens=20
#     )
#     print(f"Response: {resp_sync_non_stream.choices[0].message.content}")
#     print(f"Usage: {resp_sync_non_stream.usage}")
# except Exception as e:
#     print(f"Error in sync non-streaming: {e}")
#     traceback.print_exc()
# print("=" * 40)
#
# print("Testing asynchronous non-streaming:")
# try:
#     resp_async_non_stream = asyncio.run(
#         litellm.acompletion(
#             model=ollama_model,
#             messages=[{"role": "user", "content": "Hello from async non-stream!"}],
#         max_tokens=20
#         )
#     )
#     print(f"Response: {resp_async_non_stream.choices[0].message.content}")
#     print(f"Usage: {resp_async_non_stream.usage}")
# except Exception as e:
#     print(f"Error in async non-streaming: {e}")
#     traceback.print_exc()
# print("=" * 40)
#
# print("Testing synchronous streaming:")
# try:
#     response_sync_stream = litellm.completion(
#         model=ollama_model,
#         messages=[{"role": "user", "content": "Hello from sync stream!"}],
#         stream=True, max_tokens=20, stream_options={"include_usage": True}
#     )
#     full_response_content_sync = ""
#     usage_sync = None
#     print("Iterating over sync stream:")
#     for chunk_num, chunk in enumerate(response_sync_stream):
#         content = None
#         # For ModelResponseStream, content is in choices[0].delta.content
#         if chunk.choices and chunk.choices[0].delta:
#             content = chunk.choices[0].delta.content
#
#         finish_reason = chunk.choices[0].finish_reason
#         print(f"  Sync Stream Chunk {chunk_num + 1}: finish_reason='{finish_reason}', content='{content or ''}'")
#         if content:
#             full_response_content_sync += content
#
#         # Usage is typically on the last chunk or when finish_reason is not None
#         if hasattr(chunk, 'usage') and chunk.usage is not None:
#             usage_sync = chunk.usage
#     print(f"Full Sync Streamed Response: {full_response_content_sync}")
#     if usage_sync:
#         print(f"Sync Stream Usage: {usage_sync}")
# except Exception as e:
#     print(f"Error in sync streaming: {e}")
#     traceback.print_exc()
# print("=" * 40)
#
# print("Testing asynchronous streaming:")
#
#
# async def consume_async_stream():
#     try:
#         response_async_stream = await litellm.acompletion(
#             model=ollama_model,
#             messages=[{"role": "user", "content": "Hello from async stream!"}],
#             stream=True,
#         max_tokens=20
#         )
#         full_response_content_async = ""
#         usage_async = None
#         print("Iterating over async stream:")
#         chunk_num = 0
#         async for chunk in response_async_stream:
#             chunk_num += 1
#             content = None
#             # For ModelResponseStream, content is in choices[0].delta.content
#             if chunk.choices and chunk.choices[0].delta:
#                 content = chunk.choices[0].delta.content
#
#             finish_reason = chunk.choices[0].finish_reason
#             print(f"  Async Stream Chunk {chunk_num}: finish_reason='{finish_reason}', content='{content or ''}'")
#             if content:
#                 full_response_content_async += content
#
#             if hasattr(chunk, 'usage') and chunk.usage is not None:
#                 usage_async = chunk.usage
#         print(f"Full Async Streamed Response: {full_response_content_async}")
#         if usage_async:
#             print(f"Async Stream Usage: {usage_async}")
#     except Exception as e:
#         print(f"Error in async streaming: {e}")
#         traceback.print_exc()
#
#
# asyncio.run(consume_async_stream())
# print("=" * 40)
