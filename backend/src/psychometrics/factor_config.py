"""Factor structure configuration loader and semopy model description generator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class FactorStructure:
    """Holds the mapping of factor names to their indicator item columns."""
    factors: dict[str, list[str]]  # factor_name -> [item_col_names]
    labels: dict[str, str]         # factor_name -> human-readable label

    @property
    def factor_names(self) -> list[str]:
        return list(self.factors.keys())

    @property
    def all_items(self) -> list[str]:
        items = []
        for cols in self.factors.values():
            items.extend(cols)
        return items


def load_factor_structure(config_path: Path) -> FactorStructure:
    """Load factor structure from YAML config file."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    structure = raw["factor_structure"]

    factors: dict[str, list[str]] = {}
    labels: dict[str, str] = {}

    for factor_name, factor_data in structure.items():
        factors[factor_name] = factor_data["items"]
        labels[factor_name] = factor_data.get("label", factor_name)

    return FactorStructure(factors=factors, labels=labels)


def to_semopy_syntax(structure: FactorStructure) -> str:
    """
    Convert factor structure to semopy model description string.
    Example output:
        program_quality =~ item_1 + item_2 + item_3
        resources_teaching =~ item_4 + item_5 + item_6
        digital_assessment =~ item_7 + item_8 + item_9
    """
    lines = []
    for factor_name, items in structure.factors.items():
        items_str = " + ".join(items)
        lines.append(f"{factor_name} =~ {items_str}")
    return "\n".join(lines)
