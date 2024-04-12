import pytest
from prodpadlm_client.client_types.messages import Message, TextBlock, Usage
from prodpadlm_client.resources.api import ProdPADLM_API, MessageParam
from prodpadlm_client.client_types.messages import Message
from prodpadlm_client.client import ProdPadLMChat, _format_messages
from langchain_core.messages import HumanMessage, BaseMessage

def test_message_creation():
    text_block = TextBlock(text="Hello, world!", type="text")
    usage = Usage(input_tokens=10, output_tokens=20)
    message = Message(
        id="1234",
        content=[text_block],
        model="gpt-4",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=usage
    )

    assert message.id == "1234"
    assert message.type == "message"


def test_create():
    # Initialize the client
    client = ProdPADLM_API.Client(api_key="test_key", base_url="http://google.com")

    # Define the parameters for the create method
    params = {
        "max_tokens": 10,
        "messages": [MessageParam(content="Hello", role="user")],
        "model": "test_model",
        "stop_sequences": ["stop"],
        "stream": False,
        "system": "test_system",
        "temperature": 0.5,
        "top_k": 5,
        "top_p": 0.5,
    }

    # Call the create method
    response = client.create(**params)

    # Assert that the response is a Message instance
    assert isinstance(response, Message)


def test_format_params():
    # Initialize the ProdPadLMChat instance
    chat = ProdPadLMChat(
        prodpadlm_api_url="http://testurl.com",
        prodpadlm_api_key="test_key",
        max_tokens=1024,
        temperature=0.5,
        top_k=5,
        top_p=0.5,
        streaming=False
    )

    # Define the messages
    messages = [HumanMessage(content="Hello")]

    # Call the _format_params method
    params = chat._format_params(messages=messages, stop=["stop"])

    # Assert the parameters
    assert params["max_tokens"] == 1024
    assert params["temperature"] == 0.5
    assert params["top_k"] == 5
    assert params["top_p"] == 0.5
    assert params["stop_sequences"] == ["stop"]
    assert params["system"] is None
    assert params["messages"] == _format_messages(messages)
