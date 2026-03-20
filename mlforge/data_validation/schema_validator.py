"""
schema_validator.py
-------------------
Check that your data has the right columns, types, and value ranges.

Catches problems like:
  - A column is missing entirely
  - Age column contains negative numbers
  - Price column has text values instead of numbers

Usage:
    from mlforge.data_validation import SchemaValidator

    schema = {
        "age"   : {"type": "int",   "min": 0,   "max": 120},
        "income": {"type": "float", "min": 0},
        "name"  : {"type": "str"},
    }

    validator = SchemaValidator(schema)
    result    = validator.validate(df)

    if not result["is_valid"]:
        print("Issues found:", result["failed"])
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates column names, types, and value ranges."""

    def __init__(self, schema: dict):
        self.schema = schema
        # schema format:
        # {
        #   "column_name": {
        #       "type": "int" | "float" | "str" | "bool",
        #       "min" : minimum allowed value  (optional),
        #       "max" : maximum allowed value  (optional),
        #   }
        # }

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Run all schema checks and return a results dict.

        Returns:
            {
                "is_valid": True/False,
                "passed"  : list of checks that passed,
                "failed"  : list of problems found,
            }
        """
        results = {"is_valid": True, "passed": [], "failed": []}

        self._check_columns(df, results)
        self._check_types(df, results)
        self._check_ranges(df, results)

        if results["failed"]:
            results["is_valid"] = False

        self._print_summary(results)
        return results

    def _check_columns(self, df, results):
        for col in self.schema:
            if col in df.columns:
                results["passed"].append(f"Column '{col}' exists ✓")
            else:
                results["failed"].append(f"Column '{col}' is MISSING")

    def _check_types(self, df, results):
        type_map = {
            "int"  : ["int32", "int64"],
            "float": ["float32", "float64"],
            "str"  : ["object"],
            "bool" : ["bool"],
        }
        for col, rules in self.schema.items():
            if col not in df.columns or "type" not in rules:
                continue
            expected = rules["type"]
            actual   = str(df[col].dtype)
            if actual in type_map.get(expected, []):
                results["passed"].append(
                    f"'{col}' type is correct ({actual}) ✓"
                )
            else:
                results["failed"].append(
                    f"'{col}' type is wrong — expected {expected}, got {actual}"
                )

    def _check_ranges(self, df, results):
        for col, rules in self.schema.items():
            if col not in df.columns:
                continue
            if "min" in rules:
                n = int((df[col] < rules["min"]).sum())
                if n > 0:
                    results["failed"].append(
                        f"'{col}' has {n} values below minimum {rules['min']}"
                    )
                else:
                    results["passed"].append(f"'{col}' min value ok ✓")
            if "max" in rules:
                n = int((df[col] > rules["max"]).sum())
                if n > 0:
                    results["failed"].append(
                        f"'{col}' has {n} values above maximum {rules['max']}"
                    )
                else:
                    results["passed"].append(f"'{col}' max value ok ✓")

    def _print_summary(self, results):
        logger.info("─" * 40)
        logger.info(f"Schema Validation — "
                    f"passed: {len(results['passed'])}, "
                    f"failed: {len(results['failed'])}")
        for issue in results["failed"]:
            logger.warning(f"  ✗ {issue}")
        if not results["failed"]:
            logger.info("  All checks passed!")
        logger.info("─" * 40)
