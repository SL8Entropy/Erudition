#!/usr/bin/env python3
"""Build report.md from the deep-research JSON results.

    python wellbore-ml-gaps/generate_report.py

Reads outline.yaml (topic, item order, output_dir), fields.yaml (field categories)
and every results/*.json. Skips values containing [uncertain], fields listed in a
file's `uncertain` array, and empty values. An optional preface.md next to this
script is inserted after the title.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TOC_FIELDS = ("error_mode_targeted", "implementation_effort")
TOC_LABELS = {"error_mode_targeted": "Targets", "implementation_effort": "Effort"}
INTERNAL = {"_source_file", "uncertain"}

CATEGORY_MAPPING = {
    "basics": ["basics", "basic_info", "Basic Info"],
    "problem_formulation": ["problem_formulation", "Problem Formulation"],
    "method": ["method", "technical_features", "Method"],
    "data": ["data", "Data"],
    "results": ["results", "performance_metrics", "performance", "Results"],
    "ambiguity_handling": ["ambiguity_handling", "Ambiguity Handling"],
    "relevance_to_erudition": ["relevance_to_erudition", "Relevance to Erudition"],
}
NESTED_KEYS = {k for keys in CATEGORY_MAPPING.values() for k in keys}


def title_case(s: str) -> str:
    return s.replace("_", " ").strip().capitalize()


def slug(s: str) -> str:
    s = re.sub(r"[^\w\- ]", "", s.lower()).strip()
    return re.sub(r"\s+", "-", s)


def lookup(data: dict, field: str, category: str):
    """Top level -> mapped category sub-dict -> any nested dict."""
    if field in data:
        return data[field]
    for key in CATEGORY_MAPPING.get(category, [category]):
        sub = data.get(key)
        if isinstance(sub, dict) and field in sub:
            return sub[field]
    for v in data.values():
        if isinstance(v, dict) and field in v:
            return v[field]
    return None


def is_uncertain(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or "[uncertain]" in value
    return "[uncertain]" in json.dumps(value, ensure_ascii=False)


def fmt(value, depth: int = 0) -> str:
    if isinstance(value, dict):
        parts = [f"**{title_case(k)}**: {fmt(v, depth + 1)}" for k, v in value.items() if v not in (None, "")]
        return "<br>".join(parts) if depth == 0 else "; ".join(parts)
    if isinstance(value, list):
        if value and all(isinstance(x, dict) for x in value):
            return "<br>".join(" | ".join(f"{k}: {fmt(v, depth + 1)}" for k, v in x.items()) for x in value)
        items = [fmt(x, depth + 1) for x in value]
        if sum(len(i) for i in items) < 120:
            return ", ".join(items)
        return "<br>".join(f"- {i}" for i in items)
    text = str(value).replace("\r", "").strip()
    return text.replace("\n", "<br>")


def short(value, limit: int = 70) -> str:
    """Leading label of a long free-text field, for the TOC."""
    if value is None or is_uncertain(value):
        return "?"
    text = re.sub(r"\s+", " ", str(value)).strip()
    head = re.split(r"[;.(]| - |, via | via ", text, maxsplit=1)[0].strip()
    return head if len(head) <= limit else head[: limit - 1].rstrip() + "…"


def main() -> None:
    outline = yaml.safe_load((HERE / "outline.yaml").read_text(encoding="utf-8"))
    fields = yaml.safe_load((HERE / "fields.yaml").read_text(encoding="utf-8"))["fields"]
    out_dir = (HERE.parent / outline["execution"]["output_dir"]).resolve()

    records = {}
    for path in sorted(out_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records[str(data.get("id") or path.stem.split("_")[0])] = (path, data)
    order = [it["id"] for it in outline["items"] if it["id"] in records]
    order += [k for k in records if k not in order]
    names = {it["id"]: it["name"] for it in outline["items"]}

    defined = {f["name"] for flist in fields.values() for f in flist}
    lines = [f"# Research report: wellbore geology prediction — gaps and additions for Erudition", ""]
    lines += [f"_Topic:_ {outline['topic'].strip()}", "", f"_Items:_ {len(order)} · _generated from_ `{out_dir.name}/`", ""]
    preface = HERE / "preface.md"
    if preface.exists():
        lines += [preface.read_text(encoding="utf-8").strip(), ""]

    lines += ["## Contents", ""]
    anchors = {}
    for n, iid in enumerate(order, 1):
        data = records[iid][1]
        heading = f"{iid} {names.get(iid) or data.get('name') or iid}"
        anchors[iid] = slug(heading)
        unc = set(data.get("uncertain", []))
        summary = " | ".join(
            f"{TOC_LABELS[f]}: {'?' if f in unc else short(lookup(data, f, 'relevance_to_erudition'))}"
            for f in TOC_FIELDS
        )
        lines.append(f"{n}. [{heading}](#{anchors[iid]}) - {summary}")
    lines.append("")

    for iid in order:
        path, data = records[iid]
        unc = set(data.get("uncertain", []))
        lines += ["---", "", f"## {iid} {names.get(iid) or data.get('name') or iid}", ""]
        for cat, flist in fields.items():
            rows = []
            for f in flist:
                name = f["name"]
                if name in unc:
                    continue
                value = lookup(data, name, cat)
                if is_uncertain(value):
                    continue
                rows.append(f"| {title_case(name)} | {fmt(value).replace('|', '/')} |")
            if rows:
                lines += [f"### {title_case(cat)}", "", "| Field | Value |", "|---|---|", *rows, ""]
        extra = {k: v for k, v in data.items() if k not in defined and k not in INTERNAL and k not in NESTED_KEYS}
        if extra or unc:
            lines += ["### Other info", ""]
            for k, v in extra.items():
                if not is_uncertain(v):
                    lines.append(f"- **{title_case(k)}**: {fmt(v)}")
            if unc:
                lines.append("- **Uncertain fields (omitted above)**:")
                lines += [f"  - {u}" for u in sorted(unc)]
            lines.append("")

    report = HERE / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report} ({len(order)} items)")


if __name__ == "__main__":
    main()
