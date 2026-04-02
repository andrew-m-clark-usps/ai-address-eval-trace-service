"""Test dataset definitions for address verification evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """A single evaluation test case."""

    case_id: str
    input_address: dict[str, Any]
    expected_output: dict[str, Any]
    category: str = "general"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input_address": self.input_address,
            "expected_output": self.expected_output,
            "category": self.category,
            "description": self.description,
        }


@dataclass
class EvalDataset:
    """Collection of evaluation cases."""

    name: str
    version: str
    cases: list[EvalCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "total_cases": len(self.cases),
            "categories": list({c.category for c in self.cases}),
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_file(cls, path: str | Path) -> EvalDataset:
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)
        cases = [
            EvalCase(
                case_id=c["case_id"],
                input_address=c["input_address"],
                expected_output=c["expected_output"],
                category=c.get("category", "general"),
                description=c.get("description", ""),
            )
            for c in data.get("cases", [])
        ]
        return cls(
            name=data.get("name", "unnamed"),
            version=data.get("version", "1.0"),
            cases=cases,
        )

    @classmethod
    def default_dataset(cls) -> EvalDataset:
        """Built-in evaluation dataset covering core address scenarios."""
        cases = [
            EvalCase(
                case_id="std-001",
                input_address={
                    "street": "1600 Pennsylvania Avenue NW",
                    "city": "Washington",
                    "state": "DC",
                    "zip": "20500",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "1600 PENNSYLVANIA AVE NW",
                        "city": "WASHINGTON",
                        "state": "DC",
                        "zip": "20500",
                    },
                },
                category="standard",
                description="Known valid address - White House",
            ),
            EvalCase(
                case_id="std-002",
                input_address={
                    "street": "350 Fifth Avenue",
                    "city": "New York",
                    "state": "NY",
                    "zip": "10118",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "350 5TH AVE",
                        "city": "NEW YORK",
                        "state": "NY",
                        "zip": "10118",
                    },
                },
                category="standard",
                description="Empire State Building",
            ),
            EvalCase(
                case_id="abbr-001",
                input_address={
                    "street": "123 North Main Street",
                    "city": "Springfield",
                    "state": "Illinois",
                    "zip": "62701",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "123 N MAIN ST",
                        "city": "SPRINGFIELD",
                        "state": "IL",
                        "zip": "62701",
                    },
                },
                category="abbreviation",
                description="Full directional and suffix to abbreviation",
            ),
            EvalCase(
                case_id="abbr-002",
                input_address={
                    "street": "456 South West Boulevard Apartment 12",
                    "city": "Kansas City",
                    "state": "Missouri",
                    "zip": "64108",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "456 SW BLVD APT 12",
                        "city": "KANSAS CITY",
                        "state": "MO",
                        "zip": "64108",
                    },
                },
                category="abbreviation",
                description="Compound directional and unit designation",
            ),
            EvalCase(
                case_id="typo-001",
                input_address={
                    "street": "123 Mian Stret",
                    "city": "Springfeild",
                    "state": "IL",
                    "zip": "62701",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "123 MAIN ST",
                        "city": "SPRINGFIELD",
                        "state": "IL",
                        "zip": "62701",
                    },
                },
                category="error_correction",
                description="Typos in street and city names",
            ),
            EvalCase(
                case_id="miss-001",
                input_address={
                    "street": "742 Evergreen Terrace",
                    "city": "Springfield",
                    "state": "",
                    "zip": "",
                },
                expected_output={
                    "verified": False,
                    "standardized": {},
                },
                category="missing_data",
                description="Missing state and ZIP",
            ),
            EvalCase(
                case_id="miss-002",
                input_address={
                    "street": "",
                    "city": "Chicago",
                    "state": "IL",
                    "zip": "60601",
                },
                expected_output={
                    "verified": False,
                    "standardized": {},
                },
                category="missing_data",
                description="Missing street address",
            ),
            EvalCase(
                case_id="po-001",
                input_address={
                    "street": "PO Box 1234",
                    "city": "Anytown",
                    "state": "CA",
                    "zip": "90210",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "PO BOX 1234",
                        "city": "ANYTOWN",
                        "state": "CA",
                        "zip": "90210",
                    },
                },
                category="po_box",
                description="Standard PO Box address",
            ),
            EvalCase(
                case_id="rural-001",
                input_address={
                    "street": "RR 2 Box 45",
                    "city": "Rural Town",
                    "state": "TX",
                    "zip": "75001",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "RR 2 BOX 45",
                        "city": "RURAL TOWN",
                        "state": "TX",
                        "zip": "75001",
                    },
                },
                category="rural",
                description="Rural route address",
            ),
            EvalCase(
                case_id="mil-001",
                input_address={
                    "street": "PSC 1234 Box 5678",
                    "city": "APO",
                    "state": "AE",
                    "zip": "09001",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "PSC 1234 BOX 5678",
                        "city": "APO",
                        "state": "AE",
                        "zip": "09001",
                    },
                },
                category="military",
                description="Military APO address",
            ),
            EvalCase(
                case_id="zip4-001",
                input_address={
                    "street": "1 Infinite Loop",
                    "city": "Cupertino",
                    "state": "CA",
                    "zip": "95014-2083",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "1 INFINITE LOOP",
                        "city": "CUPERTINO",
                        "state": "CA",
                        "zip": "95014-2083",
                    },
                },
                category="zip_plus_4",
                description="ZIP+4 code validation",
            ),
            EvalCase(
                case_id="unit-001",
                input_address={
                    "street": "100 Broadway",
                    "unit": "Suite 500",
                    "city": "New York",
                    "state": "NY",
                    "zip": "10005",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "100 BROADWAY STE 500",
                        "city": "NEW YORK",
                        "state": "NY",
                        "zip": "10005",
                    },
                },
                category="unit",
                description="Suite designation standardization",
            ),
            EvalCase(
                case_id="inv-001",
                input_address={
                    "street": "99999 Nonexistent Boulevard",
                    "city": "Fakeville",
                    "state": "ZZ",
                    "zip": "00000",
                },
                expected_output={
                    "verified": False,
                    "standardized": {},
                },
                category="invalid",
                description="Completely invalid address",
            ),
            EvalCase(
                case_id="inv-002",
                input_address={
                    "street": "abc",
                    "city": "123",
                    "state": "!@#",
                    "zip": "xyz",
                },
                expected_output={
                    "verified": False,
                    "standardized": {},
                },
                category="invalid",
                description="Garbage input",
            ),
            EvalCase(
                case_id="unit-002",
                input_address={
                    "street": "1000 4th Ave",
                    "unit": "Floor 30",
                    "city": "Seattle",
                    "state": "WA",
                    "zip": "98104",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "1000 4TH AVE FL 30",
                        "city": "SEATTLE",
                        "state": "WA",
                        "zip": "98104",
                    },
                },
                category="unit",
                description="Floor designation",
            ),
            EvalCase(
                case_id="hcr-001",
                input_address={
                    "street": "HC 1 Box 200",
                    "city": "Mountain View",
                    "state": "AR",
                    "zip": "72560",
                },
                expected_output={
                    "verified": True,
                    "standardized": {
                        "street": "HC 1 BOX 200",
                        "city": "MOUNTAIN VIEW",
                        "state": "AR",
                        "zip": "72560",
                    },
                },
                category="rural",
                description="Highway contract route",
            ),
        ]
        return cls(name="default-address-eval", version="1.0", cases=cases)
