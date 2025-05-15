import litellm
import httpx
import json
import time
import uuid
from typing import Optional, Dict, Any, Iterator, Union
from pathlib import Path  # Import Path for type hinting

# Placeholder for your mTLS configuration and upstream details
# These would typically be passed to the handler's constructor or configured globally.
# For demonstration, they are defined here. Replace with your actual values.
DEFAULT_CERT_PATH = "path/to/your/client.pem"
DEFAULT_KEY_PATH = "path/to/your/client.key"
DEFAULT_CA_PATH = "path/to/your/ca.pem"

# Base URLs for your upstream services
# Example: "https://your.openai.compatible.endpoint/v1"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"  # Replace if you have a custom OpenAI-like endpoint
# Example: "http://localhost:11434"
DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"  # Replace with your Ollama instance URL

# Default model to use if "sync-stream" is passed as a suffix for OpenAI
DEFAULT_OPENAI_MODEL_FOR_SYNC_STREAM = "gpt-3.5-turbo"


class MTLSUdynamicUpstreamHandler(litellm.CustomLLM):
    """
    A LiteLLM custom handler that supports mTLS authentication and routes
    requests to different upstream LLMs (e.g., OpenAI, Ollama) based on the
    model string suffix.

    The model string is expected in the format: "mtls_prefix/actual_model_identifier"
    - "mtls_prefix/ollama_model:tag" -> routes to Ollama
    - "mtls_prefix/openai_model_name" -> routes to OpenAI compatible endpoint
    - "mtls_prefix/sync-stream" -> routes to a default OpenAI model (configurable)
    """

    def __init__(
            self,
            cert_path: Optional[Union[str, Path]] = None,  # Allow Path objects for convenience
            key_path: Optional[Union[str, Path]] = None,  # Allow Path objects for convenience
            ca_path: Optional[Union[str, Path]] = None,  # Allow Path objects for convenience
            openai_api_base: Optional[str] = None,
            ollama_api_base: Optional[str] = None,
            default_openai_model_for_sync_stream: Optional[str] = None,
    ):
        super().__init__()  # Initialize the base class

        # Determine actual paths to use, falling back to defaults
        _cert_p = cert_path or DEFAULT_CERT_PATH
        _key_p = key_path or DEFAULT_KEY_PATH
        _ca_p = ca_path or DEFAULT_CA_PATH

        # Ensure paths are strings for httpx compatibility
        # httpx expects string paths for certificates and CA bundles.
        self.cert_path = str(_cert_p) if _cert_p else None
        self.key_path = str(_key_p) if _key_p else None
        self.ca_path = str(_ca_p) if _ca_p else None

        self.openai_api_base = openai_api_base or DEFAULT_OPENAI_API_BASE
        self.ollama_api_base = ollama_api_base or DEFAULT_OLLAMA_API_BASE
        self.default_openai_model = default_openai_model_for_sync_stream or DEFAULT_OPENAI_MODEL_FOR_SYNC_STREAM

        # mTLS client configuration
        # Ensure that self.cert_path and self.key_path are not None if mTLS is intended.
        # httpx.Client cert parameter expects a (certfile, keyfile) tuple or a single certfile path.
        if self.cert_path and self.key_path:
            self.mtls_certs: Optional[tuple[str, str]] = (self.cert_path, self.key_path)
        elif self.cert_path:  # If only cert_path is provided (e.g., PKCS12 file, though httpx prefers PEM)
            self.mtls_certs: Optional[str] = self.cert_path  # type: ignore
        else:
            self.mtls_certs = None

        self.mtls_verify: Union[str, bool, None] = self.ca_path  # Can also be True/False for httpx

        # This attribute is required by litellm's CustomLLM interface
        self.model_names = ["mtls_handler"]  # A placeholder, actual routing is dynamic

    def _determine_target_and_payload(self, model_identifier: str, messages: list, stream: bool, **kwargs) -> Dict[
        str, Any]:
        """
        Determines the target upstream (Ollama or OpenAI) and prepares the
        request payload.

        Args:
            model_identifier (str): The part of the model string after the handler's prefix.
                                    e.g., "qwen3:0.6b", "gpt-4o", "sync-stream".
            messages (list): List of message objects.
            stream (bool): Whether to stream the response.
            **kwargs: Additional parameters from litellm.completion.

        Returns:
            dict: Containing "url", "payload", "provider", "model_for_stream_meta".
        """
        provider: str
        target_url: str
        payload_model: str
        # headers = {"Content-Type": "application/json"} # httpx sets this by default for json payload

        # Copy kwargs to payload, excluding LitellM internal params or params not supported by upstreams
        # This needs careful handling based on what OpenAI/Ollama support.
        # For simplicity, we'll pass common ones like max_tokens, temperature.
        payload_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ["max_tokens", "temperature", "top_p", "n", "stop", "presence_penalty", "frequency_penalty", "user",
                     "seed", "tools", "tool_choice"]  # Added seed, tools, tool_choice
        }
        if 'max_tokens' in payload_kwargs and payload_kwargs['max_tokens'] is None:
            del payload_kwargs['max_tokens']

        if model_identifier == "sync-stream":
            provider = "openai"
            payload_model = self.default_openai_model
            target_url = f"{self.openai_api_base.rstrip('/')}/chat/completions"
        elif ":" in model_identifier:  # Convention for Ollama models like "qwen2:0.5b"
            provider = "ollama"
            payload_model = model_identifier  # Ollama uses the full tag
            target_url = f"{self.ollama_api_base.rstrip('/')}/api/chat"  # Ollama's chat completions endpoint

            # Ollama specific payload adjustments for parameters inside 'options'
            # Common parameters like temperature, top_p, num_predict (max_tokens) go into options.
            ollama_options_map = {
                "max_tokens": "num_predict",
                "temperature": "temperature",
                "top_p": "top_p",
                "stop": "stop",  # stop can be a list of strings
                "seed": "seed",
                # Add other mappings if needed: "num_ctx", "repeat_penalty", etc.
            }
            current_ollama_options = payload_kwargs.pop('options', {})  # Preserve any explicitly passed options

            for litellm_key, ollama_key in ollama_options_map.items():
                if litellm_key in payload_kwargs:
                    current_ollama_options[ollama_key] = payload_kwargs.pop(litellm_key)

            if current_ollama_options:  # Only add options if there are any
                payload_kwargs['options'] = current_ollama_options

        else:  # Default to OpenAI compatible
            provider = "openai"
            payload_model = model_identifier
            target_url = f"{self.openai_api_base.rstrip('/')}/chat/completions"

        # Construct the base request payload
        request_payload = {
            "model": payload_model,
            "messages": messages,
            "stream": stream,
            **payload_kwargs,  # Add remaining kwargs that were not moved to options for Ollama
        }

        # Ensure Ollama specific structure if provider is ollama
        # (already handled by moving relevant keys into `options` above)
        # if provider == "ollama":
        #     # Example: if some top-level params for ollama are not in payload_kwargs yet
        #     # request_payload["format"] = kwargs.get("format") # if 'json' mode is needed for ollama
        #     pass

        return {
            "url": target_url,
            "payload": request_payload,
            "provider": provider,
            "model_for_stream_meta": payload_model  # Model name to embed in stream chunks
        }

    def completion(self, model: str, messages: list, **kwargs) -> litellm.utils.ModelResponse:
        """
        Handles blocking completion requests.
        `model` is the full string like "mtls_openai_llm/qwen3:0.6b".
        """
        model_identifier = model.split("/", 1)[1] if "/" in model else model

        target_info = self._determine_target_and_payload(model_identifier, messages, stream=False, **kwargs)

        request_payload = target_info["payload"]
        target_url = target_info["url"]
        provider = target_info["provider"]

        # For debugging:
        # print(f"MTLS Handler (completion): Request to {provider} at {target_url}")
        # print(f"MTLS Handler (completion): Payload: {json.dumps(request_payload, indent=2)}")
        # print(f"MTLS Handler (completion): mTLS certs: {self.mtls_certs}, verify: {self.mtls_verify}")

        with httpx.Client(cert=self.mtls_certs, verify=self.mtls_verify or True,
                          timeout=kwargs.get("request_timeout", litellm.request_timeout)) as client:
            response = client.post(target_url, json=request_payload)
            # print(f"MTLS Handler (completion): Response status: {response.status_code}") # Debug
            response.raise_for_status()  # Raise an exception for HTTP errors
            response_json = response.json()

        # print(f"MTLS Handler: Received response from {provider}: {json.dumps(response_json, indent=2)}")

        if provider == "openai":
            # LiteLLM can convert the raw response dict to ModelResponse if it's OpenAI compatible
            model_response = litellm.ModelResponse(response_json=response_json,
                                                   model=target_info["model_for_stream_meta"])
        elif provider == "ollama":
            message_obj = response_json.get("message", {})
            usage_stats = {
                "prompt_tokens": response_json.get("prompt_eval_count", 0),
                "completion_tokens": response_json.get("eval_count", 0),
                "total_tokens": response_json.get("prompt_eval_count", 0) + response_json.get("eval_count", 0),
            }
            choice = litellm.utils.Choices(
                finish_reason="stop" if response_json.get("done") else None,
                index=0,
                message=litellm.utils.Message(
                    content=message_obj.get("content"),
                    role=message_obj.get("role", "assistant")
                )
            )
            model_response = litellm.utils.ModelResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                choices=[choice],
                model=response_json.get("model", target_info["model_for_stream_meta"]),
                usage=litellm.utils.Usage(**usage_stats),
                created=int(time.mktime(time.strptime(response_json["created_at"],
                                                      "%Y-%m-%dT%H:%M:%S.%fZ"))) if "created_at" in response_json else int(
                    time.time())
            )
            model_response._hidden_params["original_response"] = response_json
        else:
            raise NotImplementedError(f"Non-streaming for provider '{provider}' not fully implemented.")

        return model_response

    def streaming(self, model: str, messages: list, **kwargs) -> Iterator[str]:
        """
        Handles streaming completion requests.
        `model` is the full string like "mtls_openai_llm/qwen3:0.6b".
        Yields OpenAI-compatible SSE chunks as strings.
        """
        model_identifier = model.split("/", 1)[1] if "/" in model else model
        stream_options = kwargs.get("stream_options", {})
        include_usage = stream_options.get("include_usage", False)

        target_info = self._determine_target_and_payload(model_identifier, messages, stream=True, **kwargs)

        request_payload = target_info["payload"]
        target_url = target_info["url"]
        provider = target_info["provider"]
        sse_model_name = target_info["model_for_stream_meta"]

        # For debugging:
        # print(f"MTLS Handler (streaming): Request to {provider} at {target_url}")
        # print(f"MTLS Handler (streaming): Payload: {json.dumps(request_payload, indent=2)}")
        # print(f"MTLS Handler (streaming): mTLS certs: {self.mtls_certs}, verify: {self.mtls_verify}")

        chunk_id = f"chatcmpl-{uuid.uuid4()}"

        with httpx.Client(cert=self.mtls_certs, verify=self.mtls_verify or True,
                          timeout=kwargs.get("request_timeout", litellm.request_timeout)) as client:
            with client.stream("POST", target_url, json=request_payload) as response:
                # print(f"MTLS Handler (streaming): Response status: {response.status_code}") # Debug
                response.raise_for_status()

                if provider == "openai":
                    for line in response.iter_lines():
                        if line:
                            yield line + "\n\n"
                elif provider == "ollama":
                    accumulated_ollama_usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                    for line in response.iter_lines():
                        if not line:
                            continue

                        try:
                            ollama_chunk = json.loads(line)
                        except json.JSONDecodeError:
                            # print(f"Warning: Could not decode JSON from Ollama stream: {line}")
                            continue

                        created_time = int(time.mktime(time.strptime(ollama_chunk["created_at"],
                                                                     "%Y-%m-%dT%H:%M:%S.%fZ"))) if "created_at" in ollama_chunk else int(
                            time.time())

                        delta_content = None
                        finish_reason = None
                        role = None  # Ollama usually sends role in the first message chunk if at all

                        if ollama_chunk.get("done") == True:
                            finish_reason = "stop"
                            # Final usage stats from Ollama
                            accumulated_ollama_usage["prompt_tokens"] = ollama_chunk.get("prompt_eval_count",
                                                                                         ollama_chunk.get(
                                                                                             "prompt_token_count", 0))
                            accumulated_ollama_usage["completion_tokens"] = ollama_chunk.get("eval_count", 0)
                            accumulated_ollama_usage["total_tokens"] = (
                                    accumulated_ollama_usage["prompt_tokens"] +
                                    accumulated_ollama_usage["completion_tokens"]
                            )
                        else:
                            message_part = ollama_chunk.get("message", {})
                            if "content" in message_part:
                                delta_content = message_part["content"]
                            if "role" in message_part:  # Capture role if present
                                role = message_part["role"]

                        choice_data = {"index": 0, "delta": {}}
                        if role:
                            choice_data["delta"]["role"] = role
                        if delta_content is not None:
                            choice_data["delta"]["content"] = delta_content

                        if finish_reason:
                            choice_data["finish_reason"] = finish_reason

                        sse_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": sse_model_name,
                            "choices": [choice_data],
                        }

                        if finish_reason and include_usage:
                            sse_chunk["usage"] = accumulated_ollama_usage

                        yield f"data: {json.dumps(sse_chunk)}\n\n"

                    # Send [DONE] message
                    yield "data: [DONE]\n\n"
                else:
                    raise NotImplementedError(f"Streaming for provider '{provider}' not implemented.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Example of instantiation and registration (you'd do this in your app setup)
if __name__ == "__main__":
    import os  # os needed for path operations if Path is not used everywhere

    # --- Certificate Paths ---
    # Assuming this script is in 'python-client' and 'certs' is a sibling directory
    # e.g., project_root/certs and project_root/python-client/dynamic_mtls_handler.py
    try:
        current_file_dir = Path(__file__).resolve().parent
        certs_dir = current_file_dir.parent / "certs"  # Moves one level up from current_file_dir to find 'certs'

        # Check if certs_dir exists, otherwise fall back to dummy paths to prevent FileNotFoundError
        # if not certs_dir.is_dir():
        #     print(f"Warning: Certs directory '{certs_dir}' not found. Using placeholder paths.")
        #     # Fallback to dummy paths if actual certs are not found for local testing
        #     # This allows the script to run without immediate file errors if certs are missing.
        #     # However, actual mTLS calls will fail if these paths are not valid.
        #     dummy_cert_dir = current_file_dir / "dummy_certs_for_litellm_handler"
        #     os.makedirs(dummy_cert_dir, exist_ok=True)
        #     CERTIFICATE_PATH_STR = str(dummy_cert_dir / "client.pem")
        #     KEY_PATH_STR = str(dummy_cert_dir / "client.key")
        #     CA_PATH_STR = str(dummy_cert_dir / "ca.pem")
        #     for p_str in [CERTIFICATE_PATH_STR, KEY_PATH_STR, CA_PATH_STR]:
        #         if not Path(p_str).exists():
        #             with open(p_str, "w") as f: f.write(f"dummy content for {Path(p_str).name}")
        # else:
        CERTIFICATE_PATH_STR = str(certs_dir / "client.crt")
        KEY_PATH_STR = str(certs_dir / "client.key")
        CA_PATH_STR = str(certs_dir / "ca.crt")

        # Ensure the paths exist for the test, or create dummy files
        # This part is crucial for the __main__ example to run without FileNotFoundError
        # if the certs are not actually present at the specified paths.
        for p_str_path in [CERTIFICATE_PATH_STR, KEY_PATH_STR, CA_PATH_STR]:
            p_obj = Path(p_str_path)
            if not p_obj.exists():
                print(f"Warning: Certificate file '{p_obj}' not found. Creating dummy file for testing.")
                os.makedirs(p_obj.parent, exist_ok=True)
                with open(p_obj, "w") as f: f.write(f"dummy cert content for {p_obj.name}")
                # For the test to proceed, these dummy files are needed.
                # Actual mTLS will fail, but the code structure can be tested.

    except NameError:  # __file__ is not defined (e.g. in some interactive environments)
        print("Warning: __file__ not defined. Using default placeholder paths for certificates.")
        CERTIFICATE_PATH_STR = DEFAULT_CERT_PATH
        KEY_PATH_STR = DEFAULT_KEY_PATH
        CA_PATH_STR = DEFAULT_CA_PATH

    # litellm.set_verbose = True

    handler_instance = MTLSUdynamicUpstreamHandler(
        cert_path=CERTIFICATE_PATH_STR,
        key_path=KEY_PATH_STR,
        ca_path=CA_PATH_STR,
        openai_api_base="http://localhost:12345/",  # Dummy, replace with actual if testing OpenAI
        ollama_api_base="http://localhost:11434"  # Standard local Ollama
    )

    LITELLM_HANDLER_PREFIX = "mtls_custom_llm"
    # Use the registration method you prefer. litellm.register_custom_provider is common.
    # litellm.register_custom_provider(LITELLM_HANDLER_PREFIX, handler_instance)
    # Or, if using custom_provider_map:
    if not hasattr(litellm, 'custom_provider_map') or litellm.custom_provider_map is None:
        litellm.custom_provider_map = []  # Initialize if it doesn't exist or is None

    # Ensure not to add duplicate providers if script is run multiple times in same session
    provider_exists = any(item.get("provider") == LITELLM_HANDLER_PREFIX for item in litellm.custom_provider_map)
    if not provider_exists:
        litellm.custom_provider_map.append(
            {"provider": LITELLM_HANDLER_PREFIX, "custom_handler": handler_instance}
        )
    else:
        print(f"Provider '{LITELLM_HANDLER_PREFIX}' already in custom_provider_map. Skipping re-addition.")

    print(f"Custom mTLS handler registered under prefix: '{LITELLM_HANDLER_PREFIX}'")
    print(f"Using certs: Cert='{CERTIFICATE_PATH_STR}', Key='{KEY_PATH_STR}', CA='{CA_PATH_STR}'")
    print("To test, ensure you have an Ollama instance running at http://localhost:11434")

    print("\n--- Example Test Calls (uncomment to run) ---")
    print(f"""
# import litellm
# litellm.set_verbose = True # Enable for detailed logs during testing

# print("\\n--- Testing Ollama Stream ---")
# try:
#     response_ollama = litellm.completion(
#         model="{LITELLM_HANDLER_PREFIX}/qwen2:0.5b", # Replace with a model you have pulled in Ollama
#         messages=[{{"role": "user", "content": "Hey, how are you? Write a short poem about robots."}}],
#         stream=True,
#         max_tokens=50, # Ollama uses options.num_predict
#         temperature=0.7,
#         stream_options={{"include_usage": True}}
#     )
#     if response_ollama:
#         full_response = ""
#         for chunk in response_ollama:
#             content = chunk.choices[0].delta.content or ""
#             full_response += content
#             print(content, end="", flush=True)
#         print("\\n--- Ollama Stream Test Done ---")
#         # print("Full Ollama response:", full_response)
#         # print("Last chunk (for usage if available):", chunk)
# except Exception as e:
#     print(f"Ollama stream test failed: {{e}}")

# print("\\n--- Testing Ollama Non-Stream ---")
# try:
#     response_ollama_non_stream = litellm.completion(
#         model="{LITELLM_HANDLER_PREFIX}/qwen2:0.5b", # Replace with a model you have pulled
#         messages=[{{"role": "user", "content": "What is the capital of France?"}}],
#         stream=False,
#         max_tokens=10
#     )
#     if response_ollama_non_stream:
#         print("Ollama Non-Stream Response:")
#         print(response_ollama_non_stream.choices[0].message.content)
#         print("Usage:", response_ollama_non_stream.usage)
#     print("--- Ollama Non-Stream Test Done ---")
# except Exception as e:
#     print(f"Ollama non-stream test failed: {{e}}")

# print("\\n--- Testing OpenAI Default (sync-stream) ---")
# # This will hit your configured DEFAULT_OPENAI_API_BASE (dummy http://localhost:12345/v1 by default)
# # Ensure a compatible service is running there if you uncomment this test.
# try:
#     response_openai_default = litellm.completion(
#         model="{LITELLM_HANDLER_PREFIX}/sync-stream", 
#         messages=[{{"role": "user", "content": "Tell me a very short joke."}}],
#         stream=True,
#         max_tokens=30,
#         stream_options={{"include_usage": True}}
#     )
#     if response_openai_default:
#         for chunk in response_openai_default:
#             print(chunk.choices[0].delta.content or "", end="", flush=True)
#     print("\\n--- OpenAI Default Test Done ---")
# except Exception as e:
#     print(f"OpenAI default test failed: {{e}} (Ensure a server is at {DEFAULT_OPENAI_API_BASE} or your configured base)")

""")
    # Quick structural test (will use dummy certs if real ones not found)
    print("\nAttempting a quick structural test call (uses a non-existent model)...")
    try:
        response = litellm.completion(
            model=f"{LITELLM_HANDLER_PREFIX}/thismodeldoesnotexist:latest",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            request_timeout=5,
            max_tokens=5
        )
        print("Structural test call response (non-stream):", response)
    except litellm.exceptions.APIConnectionError as e:
        print(
            f"Structural test call failed with APIConnectionError (expected if endpoint not reachable or mTLS issue with dummy certs): {e}")
    except httpx.ConnectError as e:
        print(
            f"Structural test call failed with httpx.ConnectError (expected if endpoint {DEFAULT_OLLAMA_API_BASE}/api/chat not reachable): {e}")
    except Exception as e:
        print(f"Structural test call failed with an unexpected error: {type(e).__name__} - {e}")

