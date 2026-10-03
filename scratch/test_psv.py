from typing import Any, Optional
from pydantic import BaseModel
from pydantic_core import core_schema

class PushStrengthValue(str):
    _legacy_alias: Optional[str] = None

    def __new__(cls, value: str, legacy_alias: Optional[str] = None):
        obj = super().__new__(cls, str(value))
        obj._legacy_alias = legacy_alias
        return obj

    @property
    def legacy_alias(self) -> Optional[str]:
        return getattr(self, "_legacy_alias", None)

    def __eq__(self, other: Any) -> bool:
        if super().__eq__(other):
            return True
        alias = getattr(self, "_legacy_alias", None)
        if alias and str(other) == str(alias):
            return True
        return False

    def __hash__(self) -> int:
        return super().__hash__()

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        return core_schema.no_info_plain_validator_function(
            lambda v: v if isinstance(v, cls) else (
                cls(v["value"], v.get("legacy_alias")) if isinstance(v, dict) else cls(str(v))
            )
        )

class TestModel(BaseModel):
    push_strength: PushStrengthValue

m = TestModel(push_strength=PushStrengthValue("none", legacy_alias="confirm_and_protect"))
print("Type:", type(m.push_strength))
print("Is 'none':", m.push_strength == "none")
print("Is 'confirm_and_protect':", m.push_strength == "confirm_and_protect")
print("In ('none', 'low'):", m.push_strength in ("none", "low"))
print("Serialized:", m.model_dump())
