import time
from typing import Optional, Dict, Any, List, Union  # Standard typing imports
from pydantic import BaseModel
from openai.types.chat import ChatCompletion  # For type hinting the input OpenAI response


# Note: The following Pydantic models (ModelResponseBase, Choices, StreamingChoices, Usage, ModelResponse)
# are defined as per your provided structure. The conversion function below will use these
# locally defined models.

class ModelResponseBase(BaseModel):
    """
    A base Pydantic model that allows extra fields, which is useful for
    handling various parameters that might come from LLM responses.
    """

    class Config:
        extra = 'allow'  # Allow any additional fields not explicitly defined


class Message(BaseModel):  # Added for completeness within Choices
    """
    Represents a message object within a choice, typically containing role and content.
    This is a simplified version; LiteLLM's actual Message can be more complex.
    """
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None  # Placeholder for tool_calls structure
    function_call: Optional[Dict[str, Any]] = None  # Placeholder for function_call structure

    class Config:
        extra = 'allow'


class Choices(BaseModel):
    """
    Represents a single choice in a non-streaming LLM response.
    It includes the reason for finishing, the index of the choice, and the message.
    """
    finish_reason: Optional[str] = None
    index: Optional[int] = None
    message: Optional[Message] = None  # Using the Message model defined above

    class Config:
        extra = 'allow'


class Delta(BaseModel):  # Added for completeness within StreamingChoices
    """
    Represents the delta (change) in content for a streaming choice.
    """
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    function_call: Optional[Dict[str, Any]] = None

    class Config:
        extra = 'allow'


class StreamingChoices(BaseModel):
    """
    Represents a single choice in a streaming LLM response chunk.
    It includes the delta (the new part of the message), finish reason, and index.
    """
    delta: Optional[Delta] = None  # Using the Delta model defined above
    finish_reason: Optional[str] = None
    index: Optional[int] = None
    logprobs: Optional[Any] = None  # Added, as it's common in OpenAI streaming

    class Config:
        extra = 'allow'


class Usage(BaseModel):
    """
    Represents token usage statistics for an LLM request (prompt, completion, total).
    """
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0

    class Config:
        extra = 'allow'


def _generate_id(prefix: str = "chatcmpl-") -> str:
    """
    Generates a unique ID, typically for chat completions, using a timestamp.
    Args:
        prefix: The prefix for the generated ID.
    Returns:
        A string representing the unique ID.
    """
    return f"{prefix}{time.time()}-{hash(time.time())}"  # Added hash for more uniqueness


class ModelResponse(ModelResponseBase):
    """
    Represents a comprehensive LLM response, adaptable for both streaming and non-streaming.
    This class is defined locally as per the user's provided structure.
    """
    id: Optional[str] = None
    choices: List[Union[Choices, StreamingChoices]]
    created: Optional[int] = None
    model: Optional[str] = None
    object: Optional[str] = None
    system_fingerprint: Optional[str] = None
    usage: Optional[Usage] = None
    stream: Optional[bool] = None  # Indicates if this is part of a stream
    stream_options: Optional[Dict[str, Any]] = None
    response_ms: Optional[int] = None
    _hidden_params: Optional[Dict[str, Any]] = None  # For internal LiteLLM use
    _response_headers: Optional[Dict[str, Any]] = None  # For storing response headers

    def __init__(
            self,
            id: Optional[str] = None,
            choices: Optional[List[Union[Dict[str, Any], Choices, StreamingChoices]]] = None,
            created: Optional[int] = None,
            model: Optional[str] = None,
            object: Optional[str] = None,  # e.g., "chat.completion" or "chat.completion.chunk"
            system_fingerprint: Optional[str] = None,
            usage: Optional[Union[Dict[str, Any], Usage]] = None,
            stream: Optional[bool] = None,
            stream_options: Optional[Dict[str, Any]] = None,
            response_ms: Optional[int] = None,
            hidden_params: Optional[Dict[str, Any]] = None,  # Maps to _hidden_params
            _response_headers: Optional[Dict[str, Any]] = None,
            **params,  # Allows for additional fields from the response
    ) -> None:

        actual_choices_list: List[Union[Choices, StreamingChoices]] = []
        if stream is True:
            # Determine object type for streaming responses
            effective_object = object if object is not None else "chat.completion.chunk"
            if choices:
                for choice_data in choices:
                    if isinstance(choice_data, StreamingChoices):
                        actual_choices_list.append(choice_data)
                    elif isinstance(choice_data, dict):
                        actual_choices_list.append(StreamingChoices(**choice_data))
                    # Add handling if choice_data is already a Pydantic model from openai sdk (unlikely here)
            else:
                actual_choices_list.append(StreamingChoices())  # Default if no choices provided
        else:
            # Determine object type for non-streaming responses
            effective_object = object if object is not None else "chat.completion"
            if choices:
                for choice_data in choices:
                    if isinstance(choice_data, Choices):
                        actual_choices_list.append(choice_data)
                    elif isinstance(choice_data, dict):
                        # Ensure 'message' within choice_data is also converted if it's a dict
                        if 'message' in choice_data and isinstance(choice_data['message'], dict):
                            choice_data['message'] = Message(**choice_data['message'])
                        actual_choices_list.append(Choices(**choice_data))
            else:
                actual_choices_list.append(Choices())  # Default if no choices provided

        effective_id = id if id is not None else _generate_id()
        effective_created = created if created is not None else int(time.time())

        effective_usage: Optional[Usage] = None
        if usage is not None:
            if isinstance(usage, dict):
                effective_usage = Usage(**usage)
            elif isinstance(usage, Usage):
                effective_usage = usage
        elif stream is None or stream is False:  # Only create default Usage for non-streaming
            effective_usage = Usage()

        # Values for super().__init__
        init_values = {
            "id": effective_id,
            "choices": actual_choices_list,
            "created": effective_created,
            "model": model,
            "object": effective_object,
            "system_fingerprint": system_fingerprint,
            "stream": stream,
            "stream_options": stream_options,
            "response_ms": response_ms,
            "_hidden_params": hidden_params,  # Note: constructor arg is hidden_params
            "_response_headers": _response_headers,
        }
        if effective_usage is not None:
            init_values["usage"] = effective_usage

        super().__init__(**init_values, **params)


def convert_openai_chat_completion_to_litellm_model_response(
        openai_response: ChatCompletion,
        response_ms: Optional[int] = None,
        hidden_params: Optional[Dict[str, Any]] = None,
        _response_headers: Optional[Dict[str, Any]] = None
) -> ModelResponse:  # Return type is the locally defined ModelResponse
    """
    Converts a non-streaming OpenAI ChatCompletion object to the locally defined ModelResponse object.

    This function processes an openai.types.chat.ChatCompletion object and maps its fields
    to the fields of the ModelResponse class defined within this file.

    Args:
        openai_response: The ChatCompletion object from the OpenAI SDK (non-streaming).
        response_ms: Optional. Response time in milliseconds.
        hidden_params: Optional. LiteLLM-specific hidden parameters.
        _response_headers: Optional. HTTP response headers.

    Returns:
        An instance of the locally defined ModelResponse.
    """

    # Convert OpenAI choices to a list of dictionaries.
    # .model_dump(exclude_none=True) helps by not including keys that are None,
    # allowing Pydantic models to use their defaults if applicable.
    parsed_choices = []
    if openai_response.choices:
        for choice in openai_response.choices:
            # choice.model_dump() will convert the entire choice, including its 'message' attribute (if Pydantic)
            parsed_choices.append(choice.model_dump(exclude_none=True))

    # Convert OpenAI usage object to a dictionary, if it exists.
    parsed_usage = openai_response.usage.model_dump(exclude_none=True) if openai_response.usage else None

    # Instantiate the locally defined ModelResponse.
    # For a "normal" (non-streaming) OpenAI ChatCompletion, 'stream' is False.
    # The 'object' field from the openai_response (e.g., "chat.completion") is passed directly.
    local_model_response = ModelResponse(  # Instantiating the ModelResponse class from this file
        id=openai_response.id,
        choices=parsed_choices,  # This will be processed by ModelResponse.__init__
        created=openai_response.created,
        model=openai_response.model,
        object=openai_response.object,
        system_fingerprint=openai_response.system_fingerprint,
        usage=parsed_usage,  # This will be processed by ModelResponse.__init__
        stream=False,  # Explicitly False for a non-streaming ChatCompletion
        response_ms=response_ms,
        hidden_params=hidden_params,  # Passed to __init__ as hidden_params
        _response_headers=_response_headers
    )

    return local_model_response

