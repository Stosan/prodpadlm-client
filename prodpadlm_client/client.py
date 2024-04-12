
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.utils import (
    build_extra_kwargs,
    convert_to_secret_str,
    get_pydantic_field_names,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator

from prodpadlm_client.resources.api import ProdPADLM_API


_message_type_lookups = {"human": "user", "ai": "assistant"}

def _merge_messages(
    messages: List[BaseMessage],
) -> List[Union[SystemMessage, AIMessage, HumanMessage]]:
    """Merge runs of human/tool messages into single human messages with content blocks."""  # noqa: E501
    merged: list = []
    for curr in messages:
        if isinstance(curr, ToolMessage):
            if isinstance(curr.content, str):
                curr = HumanMessage(
                    [
                        {
                            "type": "tool_result",
                            "content": curr.content,
                            "tool_use_id": curr.tool_call_id,
                        }
                    ]
                )
            else:
                curr = HumanMessage(curr.content)
        last = merged[-1] if merged else None
        if isinstance(last, HumanMessage) and isinstance(curr, HumanMessage):
            if isinstance(last.content, str):
                new_content: List = [{"type": "text", "text": last.content}]
            else:
                new_content = last.content
            if isinstance(curr.content, str):
                new_content.append({"type": "text", "text": curr.content})
            else:
                new_content.extend(curr.content)
            last.content = new_content
        else:
            merged.append(curr)
    return merged


def _format_messages(messages: List[BaseMessage]) -> Tuple[Optional[str], List[Dict]]:
    """Format messages for ProdPadLM."""

    """
    [
                {
                    "role": _message_type_lookups[m.type],
                    "content": [_ProdPadLMMessageContent(text=m.content).dict()],
                }
                for m in messages
            ]
    """
    system: Optional[str] = None
    formatted_messages: List[Dict] = []

    merged_messages = _merge_messages(messages)
    for i, message in enumerate(merged_messages):
        if message.type == "system":
            if i != 0:
                raise ValueError("System message must be at beginning of message list.")
            if not isinstance(message.content, str):
                raise ValueError(
                    "System message must be a string, "
                    f"instead was: {type(message.content)}"
                )
            system = message.content
            continue

        role = _message_type_lookups[message.type]
        content: Union[str, List[Dict]]

        if not isinstance(message.content, str):
            # parse as dict
            assert isinstance(
                message.content, list
            ), "ProdPadLM message content must be str or list of dicts"

            # populate content
            content = []
            for item in message.content:
                if isinstance(item, str):
                    content.append(
                        {
                            "type": "text",
                            "text": item,
                        }
                    )
                elif isinstance(item, dict):
                    if "type" not in item:
                        raise ValueError("Dict content item must have a type key")
                    elif item["type"] == "tool_use":
                        item.pop("text", None)
                        content.append(item)
                    else:
                        content.append(item)
                else:
                    raise ValueError(
                        f"Content items must be str or dict, instead was: {type(item)}"
                    )
        else:
            content = message.content

        formatted_messages.append(
            {
                "role": role,
                "content": content,
            }
        )
    return system, formatted_messages



class ProdPadLMChat(BaseChatModel):
  

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    _client: ProdPADLM_API.Client = Field(default=None)
    _async_client: ProdPADLM_API.AsyncClient = Field(default=None)

    max_tokens: int = Field(default=1024, alias="max_tokens_to_sample")
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: Optional[int] = None
    """Number of most likely tokens to consider at each step."""

    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""

    default_request_timeout: Optional[float] = None
    """Timeout for requests to prodpadlm Completion API. Default is 600 seconds."""

    prodpadlm_api_url: str

    prodpadlm_api_key: Optional[SecretStr] = None

    default_headers: Optional[Mapping[str, str]] = None
    """Headers to pass to the prodpadlm clients, will be used for every API call."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    streaming: bool = False
    """Whether to use streaming or not."""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "prodpadlm-chat"

    @root_validator(pre=True)
    def build_extra(cls, values: Dict) -> Dict:
        extra = values.get("model_kwargs", {})
        all_required_field_names = get_pydantic_field_names(cls)
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        prodpadlm_api_key = convert_to_secret_str(
            values.get("prodpadlm_api_key") or os.environ.get("prodpadlm_API_KEY") or ""
        )
        values["prodpadlm_api_key"] = prodpadlm_api_key
        api_key = prodpadlm_api_key.get_secret_value()
        api_url = (
            values.get("prodpadlm_api_url")
            or os.environ.get("prodpadlm_API_URL")
        )
        
        values["prodpadlm_api_url"] = api_url

        values["_client"] = ProdPADLM_API.Client(
            api_key=api_key,
            base_url=api_url,
            default_headers=values.get("default_headers"),
        )
     
        values["_async_client"] = ProdPADLM_API.AsyncClient(
            api_key=api_key,
            base_url=api_url,
            default_headers=values.get("default_headers"),
        )
        return values

    def _format_params(
        self,
        *,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Dict,
    ) -> Dict:
        # get system prompt if any
        system, formatted_messages = _format_messages(messages)
        rtn = {
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": stop,
            "system": system,
            **self.model_kwargs,
            **kwargs,
        }
        rtn = {k: v for k, v in rtn.items() if v is not None}

        return rtn

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        for stream in self._client.stream(**params):
            with stream as strm:
                for text in strm.text_stream:
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                    if run_manager:
                        run_manager.on_llm_new_token(text, chunk=chunk)
                    yield chunk

  

    def _format_output(self, data: Any, **kwargs: Any) -> ChatResult:
        data_dict = data.model_dump()
        content = data_dict["content"]
        llm_output = {
            k: v for k, v in data_dict.items() if k not in ("content", "role", "type")
        }
        if len(content) == 1 and content[0]["type"] == "text":
            msg = AIMessage(content=content[0]["text"])
        else:
            msg = AIMessage(content=content)
        return ChatResult(
            generations=[ChatGeneration(message=msg)],
            llm_output=llm_output,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        if self.streaming:
            stream_iter = self._stream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            return generate_from_stream(stream_iter)
        else:
            data = self._client.create(**params)
        return self._format_output(data, **kwargs)
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        else:
            data = await self._async_client.create(**params)
        return self._format_output(data, **kwargs)

 

