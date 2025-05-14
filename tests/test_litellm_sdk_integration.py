# test_litellm_integrations.py
import pytest
import litellm
from litellm import CustomLLM, completion, ModelResponse, acompletion
import warnings
import os
from dotenv import load_dotenv
import asyncio
import litellm.exceptions

# Load environment variables from .env file
load_dotenv()


# Fixture to register the custom LLM provider that returns a mock response
@pytest.fixture(scope="module")
def register_custom_llm():
    """
    Registers a custom LLM provider that returns a mock response.
    """

    class MyCustomLLM(CustomLLM):
        # Simplified constructor for ModelResponse, assuming other fields are optional or defaulted by LiteLLM
        def completion(self, *args, **kwargs) -> ModelResponse:
            # Directly create ModelResponse if all required fields for your test are in choices.message.content
            # For a more complete ModelResponse, you'd include id, created, model, etc.
            mock_choice = {"message": litellm.Message(content="Hi!", role="assistant")}
            return ModelResponse(choices=[mock_choice])  # type: ignore

        async def acompletion(self, *args, **kwargs) -> ModelResponse:
            mock_choice = {"message": litellm.Message(content="Hi from async!", role="assistant")}
            return ModelResponse(choices=[mock_choice])  # type: ignore

        def embedding(self, *args, **kwargs):
            raise NotImplementedError("Embedding not implemented for this custom LLM")

    my_custom_llm_instance = MyCustomLLM()
    original_custom_provider_map = litellm.custom_provider_map
    # Ensure current_map is initialized correctly based on original_custom_provider_map's type
    if isinstance(litellm.custom_provider_map, list):
        current_map = list(litellm.custom_provider_map)
    elif isinstance(litellm.custom_provider_map, dict):  # LiteLLM might use a dict
        current_map = litellm.custom_provider_map.copy()
    else:  # Default to a list if it's None or some other type
        current_map = []

    if isinstance(current_map, list):
        current_map.append({"provider": "my-custom-llm", "custom_handler": my_custom_llm_instance})
    elif isinstance(current_map, dict):
        current_map["my-custom-llm"] = my_custom_llm_instance  # If it's a dict mapping provider name to handler

    litellm.custom_provider_map = current_map
    yield
    litellm.custom_provider_map = original_custom_provider_map


# --- Simplified and More Robust Custom LLM for Real Ollama Proxying ---
class OllamaProxyLLMForTest(CustomLLM):
    """
    A custom LLM handler that proxies requests to a real Ollama endpoint.
    This version carefully constructs parameters for inner LiteLLM calls.
    """

    def __init__(self, ollama_model_name: str, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_model_name = ollama_model_name
        self.ollama_base_url = ollama_base_url
        super().__init__()  # Call to parent constructor is important

    def _construct_inner_params(self, **kwargs_from_outer_call) -> dict:
        """
        Helper to construct a clean dictionary of parameters for the inner LiteLLM call.
        Only includes parameters relevant to the Ollama provider.
        """
        params = {
            "model": self.ollama_model_name,
            "messages": kwargs_from_outer_call.get("messages"),
            "base_url": self.ollama_base_url,
        }

        # Define a whitelist of optional parameters that are safe to pass to Ollama
        # These are common parameters for completion calls.
        allowed_optional_params = [
            "max_tokens", "temperature", "top_p", "top_k", "stop",
            "stream", "num_predict", "format", "timeout",
            "presence_penalty", "frequency_penalty", "logit_bias", "extra_headers",
            "metadata"  # metadata can be useful
        ]

        for param_name in allowed_optional_params:
            if param_name in kwargs_from_outer_call:
                params[param_name] = kwargs_from_outer_call[param_name]

        # Ensure a default timeout if not provided
        params.setdefault('timeout', 30)

        return params

    def completion(self, *args, **kwargs) -> ModelResponse:
        # kwargs here are from the initial litellm.completion(model="ollama-proxy-for-test/...", ...) call
        params_for_inner_call = self._construct_inner_params(**kwargs)

        print(
            f"OllamaProxyLLMForTest (sync) calling inner litellm.completion with keys: {list(params_for_inner_call.keys())}")

        # For debugging, you can uncomment the line below:
        # os.environ['LITELLM_LOG'] = 'DEBUG'
        try:
            response = litellm.completion(**params_for_inner_call)
        finally:
            # If you changed LITELLM_LOG, reset it if necessary, e.g., os.environ.pop('LITELLM_LOG', None)
            pass
        return response

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        # kwargs here are from the initial litellm.acompletion(model="ollama-proxy-for-test/...", ...) call
        # These kwargs will include 'acompletion=True' from the router.
        params_for_inner_call = self._construct_inner_params(**kwargs)

        # Crucially, params_for_inner_call (due to _construct_inner_params whitelist)
        # will NOT contain the 'acompletion' key from the outer kwargs.
        # This is essential to prevent the "multiple values for keyword argument 'acompletion'" error
        # when the inner litellm.acompletion calls functools.partial.

        print(
            f"OllamaProxyLLMForTest (async) calling inner litellm.acompletion with keys: {list(params_for_inner_call.keys())}")

        # os.environ['LITELLM_LOG'] = 'DEBUG' # For debugging
        try:
            response = await litellm.acompletion(**params_for_inner_call)
        finally:
            # os.environ.pop('LITELLM_LOG', None) # Reset if changed
            pass
        return response

    def embedding(self, *args, **kwargs):
        raise NotImplementedError("Embedding not implemented for this custom LLM")


@pytest.fixture(scope="module")
def register_ollama_proxy_llm():
    ollama_model_env = os.getenv("OLLAMA_MODEL_NAME")
    if not ollama_model_env:
        pytest.skip("OLLAMA_MODEL_NAME not set in .env file. Skipping tests that require real Ollama proxy.")
        return

    proxy_instance = OllamaProxyLLMForTest(ollama_model_name=ollama_model_env)
    original_custom_provider_map = litellm.custom_provider_map

    # Handle both list and dict types for custom_provider_map
    if isinstance(litellm.custom_provider_map, list):
        current_map = list(litellm.custom_provider_map)
        # Avoid duplicate registration if tests are re-run in same session (though scope="module" should prevent this)
        if not any(entry.get("provider") == "ollama-proxy-for-test" for entry in current_map):
            current_map.append({"provider": "ollama-proxy-for-test", "custom_handler": proxy_instance})
    elif isinstance(litellm.custom_provider_map, dict):
        current_map = litellm.custom_provider_map.copy()
        current_map["ollama-proxy-for-test"] = proxy_instance
    else:  # Default to list if None or other
        current_map = [{"provider": "ollama-proxy-for-test", "custom_handler": proxy_instance}]

    litellm.custom_provider_map = current_map
    yield
    litellm.custom_provider_map = original_custom_provider_map


# --- Test Cases ---

def test_custom_llm_completion(register_custom_llm):
    """Tests the custom LLM provider with a MOCK response."""
    try:
        resp = completion(
            model="my-custom-llm/my-fake-model",
            messages=[{"role": "user", "content": "Hello world!"}],
        )
        assert resp.choices[0].message.content == "Hi!", "Custom LLM (mock) did not return the expected mock response."
    except Exception as e:
        pytest.fail(f"Custom LLM (mock) completion test failed with an exception: {e}")


def test_ollama_direct_completion():
    """Tests direct completion with an Ollama model."""
    ollama_model = os.getenv("OLLAMA_MODEL_NAME")
    if not ollama_model:
        pytest.skip("OLLAMA_MODEL_NAME not set in .env file. Skipping Ollama direct test.")
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message="Use 'content=<...>' to upload raw bytes/text content.", module="httpx._models")
        try:
            print(f"\nAttempting to use Ollama model directly: {ollama_model}")
            resp = completion(
                model=ollama_model,
                messages=[{"role": "user", "content": "Hello world from direct test!"}],
                max_tokens=5,
                base_url="http://localhost:11434",
                timeout=30
            )
            assert resp.choices[0].message.content is not None, "Ollama direct response content is None."
            assert isinstance(resp.choices[0].message.content, str), "Ollama direct response content is not a string."
            assert len(resp.choices[0].message.content) > 0, "Ollama direct response content is empty."
            print(f"Ollama Direct Response ({ollama_model}): {resp.choices[0].message.content}")
        except litellm.exceptions.APIConnectionError as e:
            pytest.fail(
                f"Ollama direct completion test failed. Could not connect to Ollama at http://localhost:11434. Error: {e}")
        except litellm.exceptions.BadRequestError as e:
            pytest.fail(
                f"Ollama direct completion test failed. BadRequestError: Model '{ollama_model}' not available? Error: {e}")
        except Exception as e:
            pytest.fail(f"Ollama direct completion test failed with an unexpected exception: {e}")


def test_litellm_proxy_completion():
    """Tests completion with a LiteLLM Proxy model."""
    litellm_proxy_model = "litellm_proxy/ollama-qwen-local"
    litellm_proxy_key = "sk-1234"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message="Use 'content=<...>' to upload raw bytes/text content.", module="httpx._models")
        try:
            resp = completion(
                model=litellm_proxy_model,
                messages=[{"role": "user", "content": "Hello world from LiteLLM Proxy test!"}],
                max_tokens=5,
                base_url="http://localhost:4000",
                api_key=litellm_proxy_key,
                timeout=30
            )
            assert resp.choices[0].message.content is not None, "LiteLLM Proxy response content is None."
            assert isinstance(resp.choices[0].message.content, str), "LiteLLM Proxy response content is not a string."
            assert len(resp.choices[0].message.content) > 0, "LiteLLM Proxy response content is empty."
            print(f"LiteLLM Proxy Response: {resp.choices[0].message.content}")
        except litellm.exceptions.APIConnectionError as e:
            pytest.fail(
                f"LiteLLM Proxy completion test failed. Could not connect to LiteLLM Proxy at http://localhost:4000. Error: {e}")
        except Exception as e:
            pytest.fail(f"LiteLLM Proxy completion test failed with an unexpected exception: {e}")


def test_custom_ollama_proxy_completion(register_ollama_proxy_llm):
    """Tests synchronous completion through the OllamaProxyLLMForTest (real Ollama call)."""
    if not os.getenv("OLLAMA_MODEL_NAME"):
        pytest.skip("Skipping custom Ollama proxy test as OLLAMA_MODEL_NAME is not set.")
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message="Use 'content=<...>' to upload raw bytes/text content.", module="httpx._models")
        try:
            print("\nAttempting custom Ollama proxy (sync) completion...")
            resp = completion(
                model="ollama-proxy-for-test/some-sync-model-id",
                # This model name is for routing to the custom handler
                messages=[{"role": "user", "content": "Why is the sky blue? Answer briefly."}],
                max_tokens=15,
                timeout=45  # Explicit timeout for the test call
            )
            assert resp.choices[0].message.content is not None, "Custom Ollama Proxy (sync) response content is None."
            assert isinstance(resp.choices[0].message.content,
                              str), "Custom Ollama Proxy (sync) response content is not a string."
            assert len(resp.choices[0].message.content) > 0, "Custom Ollama Proxy (sync) response content is empty."
            print(f"Custom Ollama Proxy (Sync) Response: {resp.choices[0].message.content}")
        except litellm.exceptions.APIConnectionError as e:
            pytest.fail(f"Custom Ollama Proxy (sync) test failed to connect to Ollama. Error: {e}")
        except litellm.exceptions.BadRequestError as e:
            pytest.fail(
                f"Custom Ollama Proxy (sync) test failed. BadRequestError: Model not available in Ollama? Error: {e}")
        except Exception as e:
            pytest.fail(f"Custom Ollama Proxy (sync) test failed with an unexpected exception: {e}")


@pytest.mark.asyncio
async def test_custom_ollama_proxy_acompletion(register_ollama_proxy_llm):
    """Tests asynchronous completion through the OllamaProxyLLMForTest (real Ollama call)."""
    if not os.getenv("OLLAMA_MODEL_NAME"):
        pytest.skip("Skipping custom Ollama proxy async test as OLLAMA_MODEL_NAME is not set.")
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message="Use 'content=<...>' to upload raw bytes/text content.", module="httpx._models")
        try:
            print("\nAttempting custom Ollama proxy (async) acompletion...")
            resp = await acompletion(
                model="ollama-proxy-for-test/some-async-model-id",  # For routing to custom handler
                messages=[{"role": "user", "content": "Tell me a fun fact about otters. Be concise."}],
                max_tokens=20,
                timeout=45  # Explicit timeout for the test call
            )
            assert resp.choices[0].message.content is not None, "Custom Ollama Proxy (async) response content is None."
            assert isinstance(resp.choices[0].message.content,
                              str), "Custom Ollama Proxy (async) response content is not a string."
            assert len(resp.choices[0].message.content) > 0, "Custom Ollama Proxy (async) response content is empty."
            print(f"Custom Ollama Proxy (Async) Response: {resp.choices[0].message.content}")
        except litellm.exceptions.APIConnectionError as e:
            pytest.fail(f"Custom Ollama Proxy (async) test failed to connect to Ollama. Error: {e}")
        except litellm.exceptions.BadRequestError as e:
            pytest.fail(
                f"Custom Ollama Proxy (async) test failed. BadRequestError: Model not available in Ollama? Error: {e}")
        except Exception as e:
            pytest.fail(f"Custom Ollama Proxy (async) test failed with an unexpected exception: {e}")

# Instructions for running tests:
# 1. Ensure OLLAMA_MODEL_NAME is in your .env file (e.g., OLLAMA_MODEL_NAME="ollama/qwen3:0.6b").
# 2. The specified model must be pulled in your Ollama instance.
# 3. Install dependencies: pip install pytest litellm python-dotenv pytest-asyncio httpx
# 4. Ensure Ollama server (http://localhost:11434) and LiteLLM Proxy (http://localhost:4000 for its test) are running.
# 5. Run: pytest -s -v tests/test_litellm_sdk_integration.py
