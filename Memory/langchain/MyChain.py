from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.llm import LLMChain
from pydantic import Extra, Field, root_validator


class MyChain(LLMChain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    input_key: list = []
    output_key: str = "response"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def my_input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]



    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        memory_keys = values["memory"].memory_variables
        input_key = values["input_key"]
        for ik in input_key:
            if ik in memory_keys:
                raise ValueError(
                    f"The input key {ik} was also found in the memory keys "
                    f"({memory_keys}) - please provide keys that don't overlap."
                )
        prompt_variables = values["prompt"].input_variables
        expected_keys = memory_keys + input_key
        if set(expected_keys) != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but got {memory_keys} as inputs from "
                f"memory, and {input_key} as the normal input key."
            )
        return values