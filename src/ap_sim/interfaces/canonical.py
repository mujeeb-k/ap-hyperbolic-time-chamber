"""Value encodings and normalization. Shared by both producers.

`encode` turns a catalog-typed Python value into its JSON form (page-model producer).
`parse_cell` turns visible cell text into the same JSON form (browser producer).
`dumps` is the canonical byte form the model receives and the parity test compares.
"""
from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation

from ap_sim.app import catalog
from ap_sim.domain.records import Money, Qty
from ap_sim.events.clock import format_time, parse_time

NULL_TEXT = "—"
_QTY = re.compile(r"^(\d+(?:\.\d+)?) (EA)$")
_MONEY = re.compile(r"^€(\d{1,3}(?:,\d{3})*|\d+)\.(\d{2})$")
_INT = re.compile(r"^(0|[1-9]\d*)$")


class ParseError(ValueError):
    pass


def normalize_text(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).split())


def _qty_value(v: Decimal) -> str:
    return format(v.normalize(), "f")


def encode(col_type: str, value):
    if value is None:
        return None
    if col_type == "qty":
        assert isinstance(value, Qty)
        return {"value": _qty_value(value.value), "unit": value.unit}
    if col_type == "money":
        assert isinstance(value, Money)
        return {"value": str(value.value.quantize(Decimal("0.01"))), "currency": value.currency}
    if col_type == "time":
        return format_time(value)
    if col_type == "int":
        return int(value)
    return normalize_text(str(value))


def parse_cell(col: catalog.Column, text: str):
    text = normalize_text(text)
    if text == NULL_TEXT:
        if not col.nullable:
            raise ParseError(f"{col.header}: null in a non-nullable column")
        return None
    kind = col.type
    try:
        if kind == "qty":
            m = _QTY.match(text)
            if not m:
                raise ParseError(f"{col.header}: not a quantity: {text!r}")
            return {"value": _qty_value(Decimal(m.group(1))), "unit": m.group(2)}
        if kind == "money":
            m = _MONEY.match(text)
            if not m:
                raise ParseError(f"{col.header}: not an amount: {text!r}")
            return {"value": f"{Decimal(m.group(1).replace(',', ''))}.{m.group(2)}", "currency": "EUR"}
        if kind == "time":
            return format_time(parse_time(text))
        if kind == "int":
            if not _INT.match(text):
                raise ParseError(f"{col.header}: not an integer: {text!r}")
            return int(text)
        if kind.startswith("enum:"):
            labels = catalog.ENUM_BY_LABEL[kind[5:]]
            if text not in labels:
                raise ParseError(f"{col.header}: unknown label {text!r}")
            return labels[text]
    except (InvalidOperation, ValueError) as exc:
        raise ParseError(str(exc)) from exc
    return text


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
