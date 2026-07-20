import json
import re
from dataclasses import dataclass, field


_BAND_NAMES = {
    5: "Fifth Band",
    4: "Fourth Band",
    3: "Third Band",
    2: "Second Band",
    1: "First Band",
    0: "0 points",
}


_LEADING_NUMBERING_RE = re.compile(
    r"^\s*(?:\d+|[IVXLCDM]+|[一二三四五六七八九十]+)[\.\)、:：]\s*",
    flags=re.IGNORECASE,
)


def _strip_leading_numbering(value):
    text = str(value).strip()
    previous = None
    while text and text != previous:
        previous = text
        text = _LEADING_NUMBERING_RE.sub("", text).strip()
    return text


def _string_rule_map(value, band_number, prefix):
    if not value:
        return {}
    if isinstance(value, dict):
        return {
            str(rule_id): _strip_leading_numbering(rule)
            for rule_id, rule in value.items()
        }
    if isinstance(value, list):
        return {
            f"{prefix}_{band_number}_{idx:03d}": _strip_leading_numbering(rule)
            for idx, rule in enumerate(value, start=1)
        }
    return {f"{prefix}_{band_number}_001": _strip_leading_numbering(value)}


@dataclass
class CanonicalBand:
    band_number: int
    minimum_score: int
    maximum_score: int
    broad_tiering_rules: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.band_number <= 5:
            raise ValueError("band number must be between 0 and 5")
        if self.minimum_score > self.maximum_score:
            raise ValueError("minimum score cannot exceed maximum score")
        self.broad_tiering_rules = _string_rule_map( # TODO: not elegant
            self.broad_tiering_rules,
            self.band_number,
            "bt",
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            band_number=int(data["band_number"]),
            minimum_score=int(data["minimum_score"]),
            maximum_score=int(data["maximum_score"]),
            broad_tiering_rules=_string_rule_map(
                data.get("broad_tiering_rules", {}),
                int(data["band_number"]),
                "bt",
            ),
        )

    def to_dict(self):
        return {
            "band_number": self.band_number,
            "minimum_score": self.minimum_score,
            "maximum_score": self.maximum_score,
            "broad_tiering_rules": dict(self.broad_tiering_rules),
        }

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def score_range_text(self):
        if self.band_number == 0:
            return "0 points"
        if self.minimum_score == self.maximum_score:
            return f"{self.minimum_score} points"
        return f"{self.minimum_score}-{self.maximum_score} points"

    def header_text(self, prefix=""):
        label = _BAND_NAMES.get(self.band_number, f"Band {self.band_number}")
        if self.band_number == 0:
            return f"{prefix}{label}:"
        return f"{prefix}{label} ({self.score_range_text()}):"

    def to_text(self):
        lines = [self.header_text()]
        lines.extend(_format_rule_lines(self.broad_tiering_rules))
        return "\n".join(lines).strip()


@dataclass
class GarBand(CanonicalBand):
    within_band_scoring_rules: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.band_number <= 5:
            raise ValueError("GAR band number must be between 1 and 5")
        self.within_band_scoring_rules = _string_rule_map(
            self.within_band_scoring_rules,
            self.band_number,
            "wb",
        )

    @classmethod
    def from_canonical_band(cls, band):
        if isinstance(band, cls):
            return band
        if isinstance(band, CanonicalBand):
            return cls(
                band_number=band.band_number,
                minimum_score=band.minimum_score,
                maximum_score=band.maximum_score,
                broad_tiering_rules={},
                within_band_scoring_rules={},
            )
        return cls.from_dict(band)

    @classmethod
    def from_dict(cls, data):
        band_number = int(data["band_number"])
        return cls(
            band_number=band_number,
            minimum_score=int(data["minimum_score"]),
            maximum_score=int(data["maximum_score"]),
            broad_tiering_rules=_string_rule_map(
                data.get("broad_tiering_rules", {}),
                band_number,
                "bt",
            ),
            within_band_scoring_rules=_string_rule_map(
                data.get("within_band_scoring_rules", {}),
                band_number,
                "wb",
            ),
        )

    def to_dict(self):
        data = super().to_dict()
        data["within_band_scoring_rules"] = dict(self.within_band_scoring_rules)
        return data


@dataclass
class RefinerOperation:
    operation: str
    section: str
    band_number: int
    content: str
    reason: str
    rule_id: str | None = None

    def __post_init__(self):
        if self.operation not in {"add", "modify"}:
            raise ValueError("refiner operation must be 'add' or 'modify'")
        if self.section not in {
            "broad_tiering_rules",
            "within_band_scoring_rules",
        }:
            raise ValueError(
                "refiner operation section must be broad_tiering_rules or "
                "within_band_scoring_rules"
            )
        if type(self.band_number) is not int or not 1 <= self.band_number <= 5:
            raise ValueError("refiner operation band_number must be an integer from 1 to 5")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("refiner operation content must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("refiner operation reason must be a non-empty string")

        self.content = self.content.strip()
        self.reason = self.reason.strip()
        if self.operation == "add":
            if self.rule_id is not None:
                raise ValueError("add operations must use a null rule_id")
        elif not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("modify operations must provide a non-empty rule_id")
        else:
            self.rule_id = self.rule_id.strip()

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("refiner operation must be a JSON object")
        expected_keys = {
            "operation",
            "section",
            "band_number",
            "rule_id",
            "content",
            "reason",
        }
        actual_keys = set(data)
        if actual_keys != expected_keys:
            missing_keys = sorted(expected_keys - actual_keys)
            unexpected_keys = sorted(actual_keys - expected_keys)
            details = []
            if missing_keys:
                details.append(f"missing keys: {', '.join(missing_keys)}")
            if unexpected_keys:
                details.append(f"unexpected keys: {', '.join(unexpected_keys)}")
            raise ValueError("invalid refiner operation fields: " + "; ".join(details))
        return cls(
            operation=data["operation"],
            section=data["section"],
            band_number=data["band_number"],
            content=data["content"],
            reason=data["reason"],
            rule_id=data["rule_id"],
        )

    def to_dict(self):
        return {
            "operation": self.operation,
            "section": self.section,
            "band_number": self.band_number,
            "rule_id": self.rule_id,
            "content": self.content,
            "reason": self.reason,
        }

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class GAR:
    canonical_bands: list[GarBand]

    @classmethod
    def from_bands(cls, canonical_bands): # Only keeps band number and range
        return cls(
            canonical_bands=[
                GarBand.from_canonical_band(band)
                for band in canonical_bands
            ],
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            canonical_bands=[GarBand.from_dict(item) for item in data["canonical_bands"]],
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

    def banding_rules_text(self):
        ordered_bands = sorted(
            self.canonical_bands,
            key=lambda band: band.band_number,
            reverse=True,
        )
        return "\n\n".join(
            _format_band_rules(
                band,
                band.broad_tiering_rules,
                empty_message="Currently no rules.",
            )
            for band in ordered_bands
        ).strip()

    def within_band_scoring_rules_text(self):
        ordered_bands = sorted(
            self.canonical_bands,
            key=lambda band: band.band_number,
            reverse=True,
        )
        return "\n\n".join(
            _format_band_rules(
                band,
                band.within_band_scoring_rules,
                header_prefix="Within ",
                empty_message="Currently no rules.",
            )
            for band in ordered_bands
        ).strip()


def gar_from_value(value):
    if isinstance(value, GAR):
        return value
    if isinstance(value, dict):
        return GAR.from_dict(value)
    return GAR.from_json(value)


def gar_to_json(value):
    return gar_from_value(value).to_json()


@dataclass
class GSR:
    general_principles: list[str] = field(default_factory=list)
    canonical_bands: list[CanonicalBand] = field(default_factory=list)

    def __post_init__(self):
        self.general_principles = [
            _strip_leading_numbering(item)
            for item in self.general_principles
            if _strip_leading_numbering(item)
        ]
        self.canonical_bands = [
            band if isinstance(band, CanonicalBand) else CanonicalBand.from_dict(band)
            for band in self.canonical_bands
        ]
        band_numbers = {band.band_number for band in self.canonical_bands}
        if band_numbers != set(range(0, 6)):
            raise ValueError("GSR must contain canonical bands 0 through 5 exactly once")

    @classmethod
    def from_dict(cls, data):
        return cls(
            general_principles=[str(item) for item in data.get("general_principles", [])],
            canonical_bands=[
                CanonicalBand.from_dict(item)
                for item in data["canonical_bands"]
            ],
        )

    def to_dict(self):
        return {
            "general_principles": list(self.general_principles),
            "canonical_bands": [band.to_dict() for band in self.canonical_bands],
        }

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def scoring_principles_text(self):
        return "\n".join(
            f"{idx}. {principle}"
            for idx, principle in enumerate(self.general_principles, start=1)
        ).strip()

    def banding_rules_text(self):
        ordered_bands = sorted(
            self.canonical_bands,
            key=lambda band: band.band_number,
            reverse=True,
        )
        return "\n\n".join(band.to_text() for band in ordered_bands).strip()

    def to_text(self):
        sections = ["I. Scoring Principles"]
        principles = self.scoring_principles_text()
        if principles:
            sections.append(principles)
        sections.append("II. Score Ranges and Requirements for Each Band")
        banding = self.banding_rules_text()
        if banding:
            sections.append(banding)
        return "\n".join(sections).strip()


def gsr_from_value(value):
    if isinstance(value, GSR):
        return value
    if isinstance(value, dict):
        return GSR.from_dict(value)
    text = str(value).strip()
    return GSR.from_json(text)


def gsr_to_text(value):
    if not value:
        return ""
    if isinstance(value, GSR):
        return value.to_text()
    if isinstance(value, dict):
        return GSR.from_dict(value).to_text()
    text = str(value).strip()
    if text.startswith("{"):
        try:
            return GSR.from_json(text).to_text()
        except Exception:
            pass
    return text


def gsr_to_json(value):
    return gsr_from_value(value).to_json()


def _format_band_rules(band, rules, header_prefix="", empty_message=None):
    lines = [band.header_text(prefix=header_prefix)]
    if rules:
        lines.extend(_format_rule_lines(rules))
    elif empty_message:
        lines.append(empty_message)
    return "\n".join(lines).strip()


def _format_rule_lines(rules):
    return [
        f"{idx}. [{rule_id}] {rule}"
        for idx, (rule_id, rule) in enumerate(rules.items(), start=1)
    ]
