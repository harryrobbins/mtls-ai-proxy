Custom API Server (Custom Format)Call your custom torch-serve / internal LLM APIs via LiteLLMinfoFor calling an openai-compatible endpoint, go here.For modifying incoming/outgoing calls on proxy, go here.Quick Startimport litellm
from litellm import CustomLLM, completion, get_llm_provider


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore

my_custom_llm = MyCustomLLM()

litellm.custom_provider_map = [ # 👈 KEY STEP - REGISTER HANDLER
        {"provider": "my-custom-llm", "custom_handler": my_custom_llm}
    ]

resp = completion(
        model="my-custom-llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
    )

assert resp.choices[0].message.content == "Hi!"
OpenAI Proxy UsageSetup your custom_handler.py fileimport litellm
from litellm import CustomLLM, completion, get_llm_provider


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore


my_custom_llm = MyCustomLLM()
Add to config.yamlIn the config below, we passpython_filename: custom_handler.pycustom_handler_instance_name: my_custom_llm. This is defined in Step 1custom_handler: custom_handler.my_custom_llmmodel_list:
  - model_name: "test-model"
    litellm_params:
      model: "openai/text-embedding-ada-002"
  - model_name: "my-custom-model"
    litellm_params:
      model: "my-custom-llm/my-model"

litellm_settings:
  custom_provider_map:
  - {"provider": "my-custom-llm", "custom_handler": custom_handler.my_custom_llm}
```bash
litellm --config /path/to/config.yaml
Test it!curl -X POST '[http://0.0.0.0:4000/chat/completions](http://0.0.0.0:4000/chat/completions)' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "my-custom-model",
    "messages": [{"role": "user", "content": "Say \"this is a test\" in JSON!"}],
}'
Expected Response{
    "id": "chatcmpl-06f1b9cd-08bc-43f7-9814-a69173921216",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Hi!",
                "role": "assistant",
                "tool_calls": null,
                "function_call": null
            }
        }
    ],
    "created": 1721955063,
    "model": "gpt-3.5-turbo",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}
Add Streaming SupportHere's a simple example of returning unix epoch seconds for both completion + streaming use-cases.s/o @Eloy Lafuente for this code example.import time
from typing import Iterator, AsyncIterator
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm import CustomLLM, completion, acompletion

class UnixTimeLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> ModelResponse:
        return completion(
            model="test/unixtime",
            mock_response=str(int(time.time())),
        )  # type: ignore

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        return await acompletion(
            model="test/unixtime",
            mock_response=str(int(time.time())),
        )  # type: ignore

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        generic_streaming_chunk: GenericStreamingChunk = {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": str(int(time.time())),
            "tool_use": None,
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }
        yield generic_streaming_chunk # type: ignore # Corrected: should yield, not return

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        generic_streaming_chunk: GenericStreamingChunk = {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": str(int(time.time())),
            "tool_use": None,
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }
        yield generic_streaming_chunk # type: ignore

unixtime = UnixTimeLLM()
Image GenerationSetup your custom_handler.py fileimport litellm
from litellm import CustomLLM
from litellm.types.utils import ImageResponse, ImageObject
import time # Added import
from typing import Any, Optional, Union # Added import
import httpx # Added import
from litellm.llms.base import AsyncHTTPHandler # Added import


class MyCustomLLM(CustomLLM):
    async def aimage_generation(self, model: str, prompt: str, model_response: ImageResponse, optional_params: dict, logging_obj: Any, timeout: Optional[Union[float, httpx.Timeout]] = None, client: Optional[AsyncHTTPHandler] = None,) -> ImageResponse:
        return ImageResponse(
            created=int(time.time()),
            data=[ImageObject(url="[https://example.com/image.png](https://example.com/image.png)")],
        )

my_custom_llm = MyCustomLLM()
Add to config.yamlIn the config below, we passpython_filename: custom_handler.pycustom_handler_instance_name: my_custom_llm. This is defined in Step 1custom_handler: custom_handler.my_custom_llmmodel_list:
  - model_name: "test-model"
    litellm_params:
      model: "openai/text-embedding-ada-002"
  - model_name: "my-custom-model"
    litellm_params:
      model: "my-custom-llm/my-model"

litellm_settings:
  custom_provider_map:
  - {"provider": "my-custom-llm", "custom_handler": custom_handler.my_custom_llm}
```bash
litellm --config /path/to/config.yaml
Test it!curl -X POST 'http://0.0.0.0:4000/v1/images/generations' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "my-custom-model",
    "prompt": "A cute baby sea otter",
}'
Expected Response{
    "created": 1721955063,
    "data": [{"url": "https://example.com/image.png"}],
}
Additional ParametersAdditional parameters are passed inside optional_params key in the completion or image_generation function.Here's how to set this:import litellm
from litellm import CustomLLM, completion, get_llm_provider


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        assert kwargs["optional_params"] == {"my_custom_param": "my-custom-param"} # 👈 CHECK HERE
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore

my_custom_llm = MyCustomLLM()

litellm.custom_provider_map = [ # 👈 KEY STEP - REGISTER HANDLER
        {"provider": "my-custom-llm", "custom_handler": my_custom_llm}
    ]

resp = completion(model="my-custom-llm/my-model", my_custom_param="my-custom-param")
Setup your custom_handler.py fileimport litellm
from litellm import CustomLLM
from litellm.types.utils import ImageResponse, ImageObject
import time # Added import
from typing import Any, Optional, Union # Added import
import httpx # Added import
from litellm.llms.base import AsyncHTTPHandler # Added import


class MyCustomLLM(CustomLLM):
    async def aimage_generation(self, model: str, prompt: str, model_response: ImageResponse, optional_params: dict, logging_obj: Any, timeout: Optional[Union[float, httpx.Timeout]] = None, client: Optional[AsyncHTTPHandler] = None,) -> ImageResponse:
        assert optional_params == {"my_custom_param": "my-custom-param"} # 👈 CHECK HERE
        return ImageResponse(
            created=int(time.time()),
            data=[ImageObject(url="https://example.com/image.png")],
        )

my_custom_llm = MyCustomLLM()
Add to config.yamlIn the config below, we passpython_filename: custom_handler.pycustom_handler_instance_name: my_custom_llm. This is defined in Step 1custom_handler: custom_handler.my_custom_llmmodel_list:
  - model_name: "test-model"
    litellm_params:
      model: "openai/text-embedding-ada-002"
  - model_name: "my-custom-model"
    litellm_params:
      model: "my-custom-llm/my-model"
      my_custom_param: "my-custom-param" # 👈 CUSTOM PARAM

litellm_settings:
  custom_provider_map:
  - {"provider": "my-custom-llm", "custom_handler": custom_handler.my_custom_llm}
```bash
litellm --config /path/to/config.yaml
Test it!curl -X POST '[http://0.0.0.0:4000/v1/images/generations](http://0.0.0.0:4000/v1/images/generations)' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "my-custom-model",
    "prompt": "A cute baby sea otter",
}'
Custom Handler Specfrom litellm.types.utils import GenericStreamingChunk, ModelResponse, ImageResponse
from typing import Iterator, AsyncIterator, Any, Optional, Union
from litellm.llms.base import BaseLLM, HTTPHandler, AsyncHTTPHandler # Added imports
import httpx # Added import


class CustomLLMError(Exception):  # use this for all your exceptions
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class CustomLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def completion(self, *args, **kwargs) -> ModelResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    def image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ImageResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ImageResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")
