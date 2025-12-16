from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


_FIRST_CAP_RE = re.compile("(.)([A-Z][a-z]+)")
_ALL_CAP_RE = re.compile("([a-z0-9])([A-Z])")


def _camel_to_snake(name: str) -> str:
    first_pass = _FIRST_CAP_RE.sub(r"\1_\2", name)
    return _ALL_CAP_RE.sub(r"\1_\2", first_pass).lower()


def _convert_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {_camel_to_snake(str(k)): _convert_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_keys(item) for item in value]
    return value


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _convert_input_keys(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            return _convert_keys(data)
        return data


class PromptRequest(CamelCaseModel):
    number: float | int | None = None
    front: bool | None = None
    prompt: dict[str, Any] | None = None
    prompt_id: str | None = None
    partial_execution_targets: Any = None
    extra_data: dict[str, Any] | None = None
    client_id: str | None = None

    def to_snake_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude_unset=True)

