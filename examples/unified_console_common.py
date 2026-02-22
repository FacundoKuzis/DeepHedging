"""
Shared helpers for unified train/compare console entrypoints.

Features:
- Config resolution from ./configs with optional prompt input.
- JSON inheritance via "extends" (string or list).
- Deep-merge of parent/child configs (child overrides parent).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_ROOT = os.path.join(ROOT_DIR, "configs")


META_KEYS = {
    "extends",
    "pipeline",
    "task",
    "tags",
    "notes",
    "description_long",
}


def _normalize_ref(config_ref: str) -> str:
    ref = str(config_ref).strip()
    if not ref:
        raise ValueError("Config reference cannot be empty.")
    if not ref.endswith(".json"):
        ref = f"{ref}.json"
    return ref


def _iter_candidate_paths(config_ref: str, task: str) -> list[str]:
    ref = _normalize_ref(config_ref)

    candidates: list[str] = []
    if os.path.isabs(ref):
        candidates.append(ref)
    else:
        # Explicit relative path from repo root
        candidates.append(os.path.normpath(os.path.join(ROOT_DIR, ref)))
        # Explicit relative path from configs root
        candidates.append(os.path.normpath(os.path.join(CONFIGS_ROOT, ref)))
        # Common shorthand: configs/runs/...
        candidates.append(os.path.normpath(os.path.join(CONFIGS_ROOT, "runs", ref)))
        # Legacy folders for backward compatibility
        if task == "train":
            candidates.append(os.path.normpath(os.path.join(ROOT_DIR, "thesis_result1_configs", "train", ref)))
            candidates.append(os.path.normpath(os.path.join(ROOT_DIR, "thesis_result1b_configs", "train", ref)))
        elif task == "compare":
            candidates.append(os.path.normpath(os.path.join(ROOT_DIR, "thesis_result1_configs", "compare", ref)))
            candidates.append(os.path.normpath(os.path.join(ROOT_DIR, "thesis_result1b_configs", "compare", ref)))
            candidates.append(os.path.normpath(os.path.join(ROOT_DIR, "thesis_result1b_configs", "option_market_compare", ref)))

    return list(dict.fromkeys(candidates))


def resolve_config_path(config_name: str | None, task: str, prompt_label: str) -> str:
    if config_name is None:
        raw = input(prompt_label).strip()
    else:
        raw = str(config_name).strip()
    if not raw:
        raise ValueError("Config name cannot be empty.")

    candidates = _iter_candidate_paths(raw, task=task)
    for path in candidates:
        if os.path.isfile(path):
            return path

    # Fallback recursive search by basename in configs/runs/**/{task}/
    basename = os.path.basename(_normalize_ref(raw))
    search_root = os.path.join(CONFIGS_ROOT, "runs")
    matches: list[str] = []
    if os.path.isdir(search_root):
        for root, _, files in os.walk(search_root):
            if basename in files:
                if os.path.basename(root).lower() == task.lower():
                    matches.append(os.path.join(root, basename))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        rels = [os.path.relpath(p, ROOT_DIR) for p in matches]
        raise ValueError(
            "Config name is ambiguous. Provide a more specific path.\n"
            + "\n".join(f"- {r}" for r in rels)
        )

    raise FileNotFoundError(
        f"Config file not found for '{raw}'. Tried:\n"
        + "\n".join(f"- {p}" for p in candidates)
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_parent_path(parent_ref: str, child_path: str) -> str:
    parent_ref = str(parent_ref).strip()
    if not parent_ref:
        raise ValueError(f"Invalid empty parent reference in {child_path}")
    if not parent_ref.endswith(".json"):
        parent_ref = f"{parent_ref}.json"

    if os.path.isabs(parent_ref):
        return parent_ref

    child_dir = os.path.dirname(child_path)
    candidate_local = os.path.normpath(os.path.join(child_dir, parent_ref))
    if os.path.isfile(candidate_local):
        return candidate_local

    candidate_configs = os.path.normpath(os.path.join(CONFIGS_ROOT, parent_ref))
    if os.path.isfile(candidate_configs):
        return candidate_configs

    candidate_repo = os.path.normpath(os.path.join(ROOT_DIR, parent_ref))
    if os.path.isfile(candidate_repo):
        return candidate_repo

    raise FileNotFoundError(
        f"Cannot resolve parent config '{parent_ref}' referenced by '{child_path}'."
    )


def _load_with_inheritance(config_path: str, stack: list[str] | None = None) -> dict[str, Any]:
    stack = stack or []
    abs_path = os.path.abspath(config_path)
    if abs_path in stack:
        chain = " -> ".join(stack + [abs_path])
        raise ValueError(f"Cyclic config inheritance detected: {chain}")

    with open(abs_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a JSON object: {abs_path}")

    parent_refs = raw.get("extends")
    if parent_refs is None:
        merged: dict[str, Any] = {}
    elif isinstance(parent_refs, str):
        parent_path = _resolve_parent_path(parent_refs, abs_path)
        merged = _load_with_inheritance(parent_path, stack + [abs_path])
    elif isinstance(parent_refs, list):
        merged = {}
        for ref in parent_refs:
            if not isinstance(ref, str):
                raise ValueError(f"'extends' list must contain strings only: {abs_path}")
            parent_path = _resolve_parent_path(ref, abs_path)
            parent_cfg = _load_with_inheritance(parent_path, stack + [abs_path])
            merged = _deep_merge(merged, parent_cfg)
    else:
        raise ValueError(f"'extends' must be string or list in {abs_path}")

    child = dict(raw)
    child.pop("extends", None)
    merged = _deep_merge(merged, child)
    return merged


def load_merged_config(config_path: str) -> tuple[str, dict[str, Any]]:
    cfg = _load_with_inheritance(config_path)
    if "run_name" not in cfg or not str(cfg.get("run_name", "")).strip():
        cfg["run_name"] = os.path.splitext(os.path.basename(config_path))[0]
    return os.path.abspath(config_path), cfg


def detect_pipeline(config: dict[str, Any]) -> str:
    pipeline = str(config.get("pipeline", "")).strip().lower()
    if pipeline:
        return pipeline

    model_family = str(config.get("model_family", "")).strip().lower()
    if model_family == "deep_hedging_result_1":
        return "result1"
    if model_family == "deep_hedging_result_1b":
        return "result1b"
    raise ValueError(
        "Cannot detect pipeline. Set 'pipeline' explicitly "
        "('result1', 'result1b', 'result1b_option_market')."
    )


def strip_meta_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in META_KEYS}


def write_temp_resolved_config(config: dict[str, Any], prefix: str = "resolved_config_") -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return path

