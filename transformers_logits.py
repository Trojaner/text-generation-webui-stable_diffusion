# Implementation taken from "outlines":
# https://github.com/outlines-dev/outlines
#
# License: Apache License 2.0:
# https://github.com/outlines-dev/outlines/blob/68b71ae810e0d6815a83df525da6d707cd4e971a/LICENSE

from typing import Optional, Type, Union
import torch
from outlines.fsm.guide import Guide, RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import adapt_tokenizer, convert_json_schema_to_str
from pydantic import BaseModel
from transformers import LogitsProcessor, PreTrainedTokenizerBase
from typing_extensions import override


class FsmLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, fsm: Guide):
        self.fsm = fsm
        self._tokenizer = tokenizer
        self._fsm_state = 0
        self._is_first_token = True

    @override
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        is_first_token = self._is_first_token
        if self._is_first_token:
            self._is_first_token = False

        mask = torch.full_like(scores, -float("inf"))

        for i in range(len(input_ids)):
            if not is_first_token:
                last_token = int(input_ids[i][-1].item())
                self._fsm_state = self.fsm.get_next_state(self._fsm_state, last_token)

            allowed_tokens = self.fsm.get_next_instruction(self._fsm_state).tokens
            mask[i][allowed_tokens] = 0

        biased_scores = scores + mask
        return biased_scores  # type: ignore

    def copy(self) -> "FsmLogitsProcessor":
        return FsmLogitsProcessor(tokenizer=self._tokenizer, fsm=self.fsm.copy())


class RegexLogitsProcessor(FsmLogitsProcessor):
    def __init__(self, regex_string: str, tokenizer: PreTrainedTokenizerBase):
        assert isinstance(tokenizer, PreTrainedTokenizerBase)

        fsm = RegexGuide(regex_string, tokenizer)
        super().__init__(tokenizer=tokenizer, fsm=fsm)


class JSONLogitsProcessor(RegexLogitsProcessor):
    def __init__(
        self,
        schema: Union[dict, Type[BaseModel], str],
        tokenizer: PreTrainedTokenizerBase,
        whitespace_pattern: Optional[str] = None,
    ):
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        tokenizer = adapt_tokenizer(tokenizer=tokenizer)
        super().__init__(regex_string=regex_string, tokenizer=tokenizer)
