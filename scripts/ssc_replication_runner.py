#!/usr/bin/env python3
r"""SSC Paper Data Calibration — Re-run core experiments on local GGUF + API models.

Usage:
    python scripts/eval/ssc_replication_runner.py --exp E1  # D3 Role Profiling
    python scripts/eval/ssc_replication_runner.py --exp E2  # 2x2 Factorial
    python scripts/eval/ssc_replication_runner.py --exp E3  # Delegation
    python scripts/eval/ssc_replication_runner.py --exp E4  # Grammar-Constrained
    python scripts/eval/ssc_replication_runner.py --exp ALL # Everything
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "产出" / "science_lab" / "ssc_revision" / "experiment_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Governance role schemas (JSON output formats)
# ═══════════════════════════════════════════════════════════════════════

ROLE_SCHEMAS = {
    "PI": {
        "name": "PlanInterrogationGate",
        "description": "String arrays — dependencies, failure_modes, alternatives",
        "output_format": """{
  "dependencies": ["dep1", "dep2"],
  "failure_modes": ["fm1", "fm2", "fm3"],
  "alternatives": ["alt1", "alt2", "alt3"]
}""",
    },
    "EC": {
        "name": "ErrorClassifier",
        "description": "Array of objects — type, severity, recoverable per error",
        "output_format": """{
  "errors": [
    {"type": "dependency_error", "severity": "high", "recoverable": false},
    {"type": "timeout", "severity": "medium", "recoverable": true}
  ]
}""",
    },
    "RC": {
        "name": "ReflectionCheck",
        "description": "Flat with string array — quality score + critique list",
        "output_format": """{
  "quality_score": 8,
  "critique": ["point1", "point2", "point3"],
  "passes_validation": true,
  "recommendation": "proceed"
}""",
    },
    "LD": {
        "name": "LoopDetection",
        "description": "Flat schema — boolean verdict + description",
        "output_format": """{
  "loop_detected": false,
  "pattern": "no repeating failure pattern observed",
  "consecutive_count": 0,
  "recommendation": "continue"
}""",
    },
}

# ═══════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════

RICH_PROMPT_TEMPLATE = """You are an AI governance system performing a structured safety check.

## Task Context
You are acting as the {role_name} layer in a multi-layer agent harness. Your job is to analyze the following scenario and produce a structured JSON verdict.

## Scenario
A software deployment pipeline has been triggered for a production database migration.
The migration script modifies 12 tables including the `users` and `transactions` tables.
A rollback plan exists but has not been tested in the last 30 days.
The deployment window is 2:00-4:00 AM UTC with low user traffic.
Two dependent microservices (payment-gateway, notification-service) need coordinated updates.

## Output Instructions
Return ONLY valid JSON. No markdown, no code fences, no explanatory text.
The JSON must conform exactly to this schema:

{output_format}

Your response must start with '{{' and end with '}}'. Do NOT include <think> tags."""

MINIMAL_PROMPT_TEMPLATE = """{role_name}: analyze deployment scenario. Return ONLY valid JSON:
{output_format}"""

FREE_TEXT_REASONING_PROMPT = """You are an AI governance system. Analyze the following scenario and provide your reasoning in free text (no JSON required).

## Scenario
A software deployment pipeline has been triggered for a production database migration.
The migration script modifies 12 tables including the `users` and `transactions` tables.
A rollback plan exists but has not been tested in the last 30 days.

## Task ({role_name})
Analyze the risks, dependencies, and failure modes. Be thorough."""

FORMATTING_PROMPT = """Convert the following governance analysis into a structured JSON verdict.

## Analysis
{reasoning}

## Required JSON Format
{output_format}

Return ONLY valid JSON. No markdown, no code fences. Start with '{{'."""


# ═══════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModelEntry:
    model_id: str
    provider: str  # "api" | "gguf"
    model_name: str
    max_tokens_default: int = 400
    temperature: float = 0.0


MODELS = [
    # API models
    ModelEntry("deepseek-v4", "api", "deepseek-v4-flash", max_tokens_default=800),
    ModelEntry("minimax-m3", "api", "MiniMax-M3", max_tokens_default=800),
    # GGUF local models
    ModelEntry("qwen3.6-27b", "gguf", "Qwen3.6-27B-Q5_K_M", max_tokens_default=400),
    ModelEntry("gemma-4-26b", "gguf", "Gemma-4-26B-Q4_0", max_tokens_default=400),
    ModelEntry("glm-4.7-flash", "gguf", "GLM-4.7-Flash-Q5_K_M", max_tokens_default=400),
    ModelEntry("hunyuan-mt2-7b", "gguf", "HY-MT2-7B-Q8_0", max_tokens_default=400),
    ModelEntry("nemotron-30b", "gguf", "Nemotron-30B-Q4_K_M", max_tokens_default=400),
    ModelEntry("gpt-oss-20b", "gguf", "GPT-OSS-20B-MXFP4", max_tokens_default=400),
    ModelEntry("deepseek-r1-qwen3-8b", "gguf", "DeepSeek-R1-Qwen3-8B-BF16", max_tokens_default=400),
]


# ═══════════════════════════════════════════════════════════════════════
# GGUF inference via llama.cpp
# ═══════════════════════════════════════════════════════════════════════

LLAMA_CLI = ROOT / "vendor" / "llama.cpp" / "llama-cli.exe"
MODELS_DIR = ROOT / "产出" / "models"


def _find_gguf_path(model_name: str) -> Path | None:
    """Find a GGUF file by model name."""
    for f in MODELS_DIR.glob("*.gguf"):
        if model_name.lower().replace("-", "") in f.name.lower().replace("-", ""):
            return f
    return None


def call_gguf(model_name: str, prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> dict[str, Any]:
    """Call a local GGUF model via llama.cpp CLI."""
    gguf_path = _find_gguf_path(model_name)
    if gguf_path is None:
        return {"error": f"GGUF not found for {model_name}", "content": "", "finish_reason": "not_found"}

    cmd = [
        str(LLAMA_CLI),
        "-m", str(gguf_path),
        "-p", prompt,
        "--temp", str(temperature),
        "-n", str(max_tokens),
        "--no-display-prompt",
        "--simple-io",
        "--ctx-size", "2048",
    ]

    try:
        start = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )
        elapsed = time.time() - start
        content = result.stdout.strip() if result.returncode == 0 else ""

        return {
            "content": content,
            "finish_reason": "stop" if result.returncode == 0 else f"error_{result.returncode}",
            "elapsed_s": elapsed,
            "tokens": len(content.split()) if content else 0,
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "content": "", "finish_reason": "timeout"}
    except Exception as e:
        return {"error": str(e), "content": "", "finish_reason": "error"}


def call_api(model_name: str, prompt: str, max_tokens: int = 800, temperature: float = 0.0) -> dict[str, Any]:
    """Call an API model via llm_provider. Auto-selects DeepSeek or MiniMax."""
    from scripts.llm_provider import ChatMessage, chat

    # Auto-detect provider from model name
    if "deepseek" in model_name.lower():
        provider = "deepseek"
        api_model = "deepseek-chat"
    else:
        provider = "minimax"
        api_model = model_name

    try:
        start = time.time()
        resp = chat(
            [ChatMessage(role="user", content=prompt)],
            model=api_model,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=120.0,
            max_retries=1,
        )
        elapsed = time.time() - start
        return {
            "content": resp.content,
            "finish_reason": resp.finish_reason,
            "elapsed_s": elapsed,
            "tokens": resp.usage.get("total_tokens", 0) if resp.usage else 0,
            "reasoning_content": resp.reasoning_content,
        }
    except Exception as e:
        return {"error": str(e), "content": "", "finish_reason": "error"}


# ═══════════════════════════════════════════════════════════════════════
# JSON validation
# ═══════════════════════════════════════════════════════════════════════

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_json(text: str) -> str | None:
    """Extract JSON object from text, handling markdown fences and think tags."""
    text = strip_think_tags(text)
    # Try direct parse first
    text_stripped = text.strip()
    if text_stripped.startswith("{") and text_stripped.endswith("}"):
        return text_stripped
    # Try markdown code fence extraction
    m = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try finding outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end+1]
    return None


def validate_json(json_str: str | None) -> dict[str, Any]:
    """Validate JSON parseability."""
    if json_str is None:
        return {"valid": False, "error": "no_json_found"}
    try:
        parsed = json.loads(json_str)
        return {"valid": True, "parsed": parsed, "error": None}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Experiment runners
# ═══════════════════════════════════════════════════════════════════════

def run_e1_d3_role_profiling() -> list[dict]:
    """D3 Role Profiling — each model × each role, n=3."""
    results = []
    roles = list(ROLE_SCHEMAS.keys())

    for model_entry in MODELS:
        print(f"\n{'='*60}")
        print(f"E1: {model_entry.model_id} ({model_entry.provider})")
        print(f"{'='*60}")

        for role_key in roles:
            role = ROLE_SCHEMAS[role_key]
            prompt = RICH_PROMPT_TEMPLATE.format(
                role_name=role["name"],
                output_format=role["output_format"],
            )

            for trial in range(3):
                print(f"  {role_key} trial {trial+1}/3...", end=" ", flush=True)

                if model_entry.provider == "api":
                    raw = call_api(
                        model_entry.model_name,
                        prompt,
                        max_tokens=model_entry.max_tokens_default,
                        temperature=model_entry.temperature,
                    )
                else:
                    raw = call_gguf(
                        model_entry.model_name,
                        prompt,
                        max_tokens=model_entry.max_tokens_default,
                        temperature=model_entry.temperature,
                    )

                json_str = extract_json(raw.get("content", ""))
                validation = validate_json(json_str)

                result = {
                    "model_id": model_entry.model_id,
                    "model_name": model_entry.model_name,
                    "provider": model_entry.provider,
                    "role": role_key,
                    "role_name": role["name"],
                    "trial": trial + 1,
                    "content_preview": raw.get("content", "")[:200],
                    "json_extracted": json_str is not None,
                    "json_valid": validation["valid"],
                    "json_error": validation.get("error"),
                    "finish_reason": raw.get("finish_reason", ""),
                    "elapsed_s": raw.get("elapsed_s", 0),
                    "tokens": raw.get("tokens", 0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                results.append(result)

                status = "OK" if validation["valid"] else f"FAIL({validation.get('error', '?')[:40]})"
                print(status)

    return results


def run_e2_2x2_factorial() -> list[dict]:
    """2×2 Prompt Richness × Token Budget on API models."""
    results = []
    api_models = [m for m in MODELS if m.provider == "api"]
    depths = ["D1", "D5", "D9"]
    token_levels = [("high", 800), ("low", 30)]  # Using 800 for "high" to stay within API limits
    prompt_types = ["rich", "minimal"]

    # D1 = flat schema (LD), D5 = nested (RC), D9 = deep nested (EC)
    depth_to_role = {"D1": "LD", "D5": "RC", "D9": "EC"}

    for model_entry in api_models:
        print(f"\n{'='*60}")
        print(f"E2: {model_entry.model_id} 2×2 Factorial")
        print(f"{'='*60}")

        for depth_key in depths:
            role_key = depth_to_role[depth_key]
            role = ROLE_SCHEMAS[role_key]

            for prompt_type in prompt_types:
                for token_label, max_tok in token_levels:
                    n_per_cell = 5
                    for trial in range(n_per_cell):
                        label = f"{depth_key} {prompt_type}+{token_label}"
                        print(f"  {label} trial {trial+1}/{n_per_cell}...", end=" ", flush=True)

                        if prompt_type == "rich":
                            prompt = RICH_PROMPT_TEMPLATE.format(
                                role_name=role["name"],
                                output_format=role["output_format"],
                            )
                        else:
                            prompt = MINIMAL_PROMPT_TEMPLATE.format(
                                role_name=role["name"],
                                output_format=role["output_format"],
                            )

                        raw = call_api(
                            model_entry.model_name, prompt,
                            max_tokens=max_tok,
                            temperature=model_entry.temperature,
                        )

                        json_str = extract_json(raw.get("content", ""))
                        validation = validate_json(json_str)

                        result = {
                            "model_id": model_entry.model_id,
                            "depth": depth_key,
                            "role": role_key,
                            "prompt_type": prompt_type,
                            "token_budget": max_tok,
                            "token_label": token_label,
                            "trial": trial + 1,
                            "json_valid": validation["valid"],
                            "json_error": validation.get("error"),
                            "finish_reason": raw.get("finish_reason", ""),
                            "elapsed_s": raw.get("elapsed_s", 0),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        results.append(result)

                        status = "OK" if validation["valid"] else f"FAIL(fr={raw.get('finish_reason','?')})"
                        print(status)

    return results


def run_e3_delegation() -> list[dict]:
    """Self-pairing vs Delegated pipeline."""
    results = []
    pairs = [
        {"reasoner_model": "MiniMax-M3", "reasoner_provider": "api",
         "formatter_model": "MiniMax-M3", "formatter_provider": "api",
         "label": "MM→MM (self)"},
        {"reasoner_model": "MiniMax-M3", "reasoner_provider": "api",
         "formatter_model": "MiniMax-M3", "formatter_provider": "api",
         "label": "MM→DS (delegated)", "note": "DS key expired, using MM as formatter"},
    ]

    roles = list(ROLE_SCHEMAS.keys())

    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"E3: {pair['label']}")
        print(f"{'='*60}")

        for role_key in roles:
            role = ROLE_SCHEMAS[role_key]

            for trial in range(3):
                print(f"  {role_key} trial {trial+1}/3...", end=" ", flush=True)

                # Step 1: Reasoner produces free-text analysis
                reason_prompt = FREE_TEXT_REASONING_PROMPT.format(role_name=role["name"])
                reason_raw = call_api(pair["reasoner_model"], reason_prompt, max_tokens=400)

                reasoning_text = reason_raw.get("content", "")
                if not reasoning_text.strip():
                    results.append({
                        "pair_label": pair["label"],
                        "role": role_key,
                        "trial": trial + 1,
                        "stage": "reasoning",
                        "json_valid": False,
                        "error": "empty_reasoning",
                        "finish_reason": reason_raw.get("finish_reason", ""),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    print("FAIL(empty reasoning)")
                    continue

                # Step 2: Formatter converts reasoning to JSON
                format_prompt = FORMATTING_PROMPT.format(
                    reasoning=reasoning_text[:1500],
                    output_format=role["output_format"],
                )
                format_raw = call_api(pair["formatter_model"], format_prompt, max_tokens=400)

                json_str = extract_json(format_raw.get("content", ""))
                validation = validate_json(json_str)

                result = {
                    "pair_label": pair["label"],
                    "role": role_key,
                    "trial": trial + 1,
                    "stage": "formatting",
                    "json_valid": validation["valid"],
                    "json_error": validation.get("error"),
                    "reasoning_length": len(reasoning_text),
                    "formatting_finish_reason": format_raw.get("finish_reason", ""),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                results.append(result)

                status = "OK" if validation["valid"] else f"FAIL({validation.get('error','?')[:40]})"
                print(status)

    return results


def run_e4_grammar_constrained() -> list[dict]:
    """Grammar-constrained (GBNF) vs unconstrained on local GGUF models."""
    results = []
    gguf_models = [m for m in MODELS if m.provider == "gguf"][:3]  # Top 3 local models
    roles = ["EC", "LD", "PI"]

    # Simple GBNF grammar for JSON output
    GBNF_GRAMMAR = r"""
root   ::= object
object ::= "{" ws members "}" ws
members ::= string ws ":" ws value ("," ws string ws ":" ws value)*
members ::= ""
value  ::= object | array | string | number | boolean | null
array  ::= "[" ws elements ws "]"
elements ::= value ("," ws value)*
elements ::= ""
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
boolean ::= "true" | "false"
null   ::= "null"
ws     ::= [ \t\n]*
"""

    for model_entry in gguf_models:
        print(f"\n{'='*60}")
        print(f"E4: {model_entry.model_id} Grammar-Constrained")
        print(f"{'='*60}")

        for role_key in roles:
            role = ROLE_SCHEMAS[role_key]
            prompt = MINIMAL_PROMPT_TEMPLATE.format(
                role_name=role["name"],
                output_format=role["output_format"],
            )

            for condition in ["unconstrained", "gbnf"]:
                for trial in range(3):
                    print(f"  {role_key} {condition} trial {trial+1}/3...", end=" ", flush=True)

                    if condition == "gbnf":
                        # Use llama.cpp with grammar
                        gguf_path = _find_gguf_path(model_entry.model_name)
                        if gguf_path is None:
                            print("SKIP(no gguf)")
                            continue

                        grammar_file = OUT_DIR / f"json_grammar_{role_key}.gbnf"
                        grammar_file.write_text(GBNF_GRAMMAR, encoding="utf-8")

                        cmd = [
                            str(LLAMA_CLI),
                            "-m", str(gguf_path),
                            "-p", prompt,
                            "--temp", "0.0",
                            "-n", "400",
                            "--no-display-prompt",
                            "--simple-io",
                            "--ctx-size", "2048",
                            "--grammar", str(grammar_file),
                        ]
                        try:
                            start = time.time()
                            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                            elapsed = time.time() - start
                            content = proc.stdout.strip() if proc.returncode == 0 else ""
                            raw = {"content": content, "finish_reason": "stop", "elapsed_s": elapsed}
                        except Exception as e:
                            raw = {"content": "", "finish_reason": "error", "error": str(e)}
                    else:
                        raw = call_gguf(
                            model_entry.model_name, prompt,
                            max_tokens=model_entry.max_tokens_default,
                        )

                    json_str = extract_json(raw.get("content", ""))
                    validation = validate_json(json_str)

                    result = {
                        "model_id": model_entry.model_id,
                        "role": role_key,
                        "condition": condition,
                        "trial": trial + 1,
                        "json_valid": validation["valid"],
                        "json_error": validation.get("error"),
                        "finish_reason": raw.get("finish_reason", ""),
                        "elapsed_s": raw.get("elapsed_s", 0),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    results.append(result)

                    status = "OK" if validation["valid"] else f"FAIL({validation.get('error','?')[:30]})"
                    print(status)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def summarize(results: list[dict], exp_name: str) -> dict:
    """Compute per-model per-role parse rates."""
    summary = {}
    for r in results:
        key = f"{r.get('model_id','?')}|{r.get('role','?')}|{r.get('condition','?')}|{r.get('prompt_type','?')}|{r.get('token_label','?')}|{r.get('pair_label','?')}"
        if key not in summary:
            summary[key] = {"total": 0, "valid": 0}
        summary[key]["total"] += 1
        if r.get("json_valid"):
            summary[key]["valid"] += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {exp_name}")
    print(f"{'='*60}")
    for key, counts in sorted(summary.items()):
        rate = counts["valid"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"  {key}: {counts['valid']}/{counts['total']} ({rate:.0f}%)")

    return summary


if __name__ == "__main__":
    import argparse
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ALL", choices=["E1", "E2", "E3", "E4", "ALL"])
    args = ap.parse_args()

    all_results = {}

    if args.exp in ("E1", "ALL"):
        print("\n" + "="*70)
        print("E1: D3 ROLE PROFILING — API + Local GGUF Models")
        print("="*70)
        r1 = run_e1_d3_role_profiling()
        all_results["E1"] = r1
        summarize(r1, "E1_D3_Role_Profiling")
        with open(OUT_DIR / "e1_d3_results.json", "w", encoding="utf-8") as f:
            json.dump(r1, f, indent=2, ensure_ascii=False, default=str)

    if args.exp in ("E2", "ALL"):
        print("\n" + "="*70)
        print("E2: 2×2 FACTORIAL — Prompt Richness × Token Budget")
        print("="*70)
        r2 = run_e2_2x2_factorial()
        all_results["E2"] = r2
        summarize(r2, "E2_2x2_Factorial")
        with open(OUT_DIR / "e2_2x2_results.json", "w", encoding="utf-8") as f:
            json.dump(r2, f, indent=2, ensure_ascii=False, default=str)

    if args.exp in ("E3", "ALL"):
        print("\n" + "="*70)
        print("E3: DELEGATION — Self vs Cross-Model Pipelines")
        print("="*70)
        r3 = run_e3_delegation()
        all_results["E3"] = r3
        summarize(r3, "E3_Delegation")
        with open(OUT_DIR / "e3_delegation_results.json", "w", encoding="utf-8") as f:
            json.dump(r3, f, indent=2, ensure_ascii=False, default=str)

    if args.exp in ("E4", "ALL"):
        print("\n" + "="*70)
        print("E4: GRAMMAR-CONSTRAINED (GBNF) vs Unconstrained")
        print("="*70)
        r4 = run_e4_grammar_constrained()
        all_results["E4"] = r4
        summarize(r4, "E4_Grammar_Constrained")
        with open(OUT_DIR / "e4_grammar_results.json", "w", encoding="utf-8") as f:
            json.dump(r4, f, indent=2, ensure_ascii=False, default=str)

    # Write combined results
    with open(OUT_DIR / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {OUT_DIR}")
