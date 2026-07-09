import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CanonicalBand:
    band_number: int
    minimum_score: int
    maximum_score: int
    broad_tiering_rules: dict[str, str] = field(default_factory=dict)
    within_band_scoring_rules: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not 1 <= self.band_number <= 5:
            raise ValueError("band number must be between 1 and 5")
        if self.minimum_score > self.maximum_score:
            raise ValueError("minimum score cannot exceed maximum score")
        self.broad_tiering_rules = {
            str(rule_id): str(rule)
            for rule_id, rule in self.broad_tiering_rules.items()
        }
        self.within_band_scoring_rules = {
            str(rule_id): str(rule)
            for rule_id, rule in self.within_band_scoring_rules.items()
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            band_number=int(data["band_number"]),
            minimum_score=int(data["minimum_score"]),
            maximum_score=int(data["maximum_score"]),
            broad_tiering_rules={
                str(rule_id): str(rule)
                for rule_id, rule in data.get("broad_tiering_rules", {}).items()
            },
            within_band_scoring_rules={
                str(rule_id): str(rule)
                for rule_id, rule in data.get("within_band_scoring_rules", {}).items()
            },
        )

    def to_dict(self):
        return {
            "band_number": self.band_number,
            "minimum_score": self.minimum_score,
            "maximum_score": self.maximum_score,
            "broad_tiering_rules": dict(self.broad_tiering_rules),
            "within_band_scoring_rules": dict(self.within_band_scoring_rules),
        }

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class RefinerOperation:
    operation: str
    section: str
    content: Any
    band_number: int | None = None
    rule_id: str | None = None

    @classmethod
    def from_dict(cls, data):
        return cls(
            operation=str(data["operation"]),
            section=str(data["section"]),
            band_number=int(data["band_number"]) if data.get("band_number") is not None else None,
            rule_id=str(data["rule_id"]) if data.get("rule_id") is not None else None,
            content=data["content"],
        )

    def to_dict(self):
        return {
            "operation": self.operation,
            "section": self.section,
            "band_number": self.band_number,
            "rule_id": self.rule_id,
            "content": self.content,
        }

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class GAR:
    canonical_bands: list[CanonicalBand]

    @classmethod
    def from_bands(cls, canonical_bands):
        return cls(
            canonical_bands=list(canonical_bands),
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            canonical_bands=[CanonicalBand.from_dict(item) for item in data["canonical_bands"]],
        )

    def to_dict(self):
        return {
            "canonical_bands": [band.to_dict() for band in self.canonical_bands],
        }

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def gar_from_value(value):
    if isinstance(value, GAR):
        return value
    if isinstance(value, dict):
        return GAR.from_dict(value)
    return GAR.from_json(value)


def gar_to_json(value):
    return gar_from_value(value).to_json()
