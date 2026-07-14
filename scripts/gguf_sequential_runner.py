#!/usr/bin/env python3
r"""Sequential GGUF runner via llama-server HTTP API.

For each GGUF model in 产出/models/:
  1. taskkill old llama-server
  2. Start llama-server with target GGUF
  3. Poll /v1/models until ready
  4. Run D3 governance role profiling via HTTP POST
  5. Kill server, move to next model

Usage:
    python scripts/eval/gguf_sequential_runner.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LLAMA_SERVER = ROOT / "vendor" / "llama.cpp" / "llama-server.exe"
MODELS_DIR = ROOT / "产出" / "models"
OUT_DIR = ROOT / "产出" / "science_lab" / "ssc_revision" / "experiment_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "http://localhost:8081"
MODELS_URL = f"{API_BASE}/v1/models"
CHAT_URL = f"{API_BASE}/v1/chat/completions"

# ═══════════════════════════════════════════════════════════════════════
# D3 governance roles with minimal JSON prompts
# ═══════════════════════════════════════════════════════════════════════

ROLES: dict[str, tuple[str, str]] = {
    "PI": (
        "PlanInterrogationGate",
        'Return ONLY valid JSON, no markdown:\n{"dependencies":["dep1","dep2"],"failure_modes":["fm1","fm2","fm3"],"alternatives":["alt1","alt2","alt3"]}\n\nScenario: A production DB migration modifies 12 tables including users and transactions. Rollback plan exists but untested for 30 days. Identify dependencies, failure modes, and alternatives.',
    ),
    "EC": (
        "ErrorClassifier",
        'Return ONLY valid JSON, no markdown:\n{"errors":[{"type":"dependency_error","severity":"high","recoverable":false},{"type":"timeout","severity":"medium","recoverable":true}]}\n\nScenario: Classify errors in a deployment that failed due to untested rollback and missing service coordination.',
    ),
    "RC": (
        "ReflectionCheck",
        'Return ONLY valid JSON, no markdown:\n{"quality_score":8,"critique":["point1","point2","point3"],"passes_validation":true,"recommendation":"proceed"}\n\nScenario: Evaluate the quality of a migration plan review.',
    ),
    "LD": (
        "LoopDetection",
        'Return ONLY valid JSON, no markdown:\n{"loop_detected":false,"pattern":"no repeating failure pattern","consecutive_count":0,"recommendation":"continue"}\n\nScenario: Check if the current deployment attempt is repeating a prior failure pattern.',
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def kill_server() -> None:
    subprocess.run(
        ["taskkill", "/F", "/IM", "llama-server.exe"],
        capture_output=True,
        timeout=10,
    )


def start_server(model_path: Path, port: int = 8081) -> subprocess.Popen:
    kill_server()
    time.sleep(1)
    proc = subprocess.Popen(
        [
            str(LLAMA_SERVER),
            "-m", str(model_path),
            "--port", str(port),
            "-ngl", "99",       # offload all layers to GPU (Vulkan)
            "--ctx-size", "2048",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_until_ready(timeout: float = 180.0) -> bool:
    """Poll /v1/models until the server responds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(MODELS_URL)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def chat_completion(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> dict[str, Any]:
    """Send a chat completion request to the llama-server."""
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "content": "", "finish_reason": "exception"}

    content = ""
    finish_reason = ""
    if "choices" in raw and raw["choices"]:
        choice = raw["choices"][0]
        content = choice.get("message", {}).get("content", "") or ""
        finish_reason = choice.get("finish_reason", "")

    return {"content": content, "finish_reason": finish_reason, "raw": raw}


def extract_json(text: str) -> str | None:
    """Extract JSON object from model output."""
    text = re.sub(r"<[^>]+>.*?</[^>]+>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```", "", text)
    text = text.strip()
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        return text[s : e + 1]
    return None


def validate_json(json_str: str | None) -> tuple[bool, str]:
    if json_str is None:
        return False, "no_json_found"
    try:
        json.loads(json_str)
        return True, ""
    except json.JSONDecodeError as e:
        return False, str(e)[:80]


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> int:
    if not LLAMA_SERVER.exists():
        print(f"ERROR: llama-server.exe not found at {LLAMA_SERVER}")
        return 1

    # Find all GGUF files, sorted by size (smallest first)
    gguf_files = sorted(MODELS_DIR.glob("*.gguf"), key=lambda f: f.stat().st_size)
    if not gguf_files:
        print(f"ERROR: No .gguf files found in {MODELS_DIR}")
        return 1

    print(f"Found {len(gguf_files)} GGUF models:")
    for f in gguf_files:
        print(f"  {f.name} ({f.stat().st_size / (1024**3):.1f} GB)")

    results: list[dict] = []

    for gguf_path in gguf_files:
        model_name = gguf_path.stem
        size_gb = gguf_path.stat().st_size / (1024**3)
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} ({size_gb:.1f} GB)")
        print(f"{'='*60}")

        # Step 1-2: Start server + wait for ready
        print("  Starting llama-server...", end=" ", flush=True)
        proc = start_server(gguf_path)
        print("waiting for model to load...", end=" ", flush=True)

        if not wait_until_ready(timeout=180):
            print("TIMEOUT (model failed to load)")
            kill_server()
            # Record all roles as failed
            for role_key in ROLES:
                results.append({
                    "model": model_name, "role": role_key,
                    "json_valid": False, "json_error": "server_timeout",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            continue
        print("READY")

        # Step 3: Run D3 role profiling
        for role_key, (role_name, prompt) in ROLES.items():
            print(f"  {role_key}...", end=" ", flush=True)

            try:
                resp = chat_completion(prompt, max_tokens=256, temperature=0.0)
            except Exception as e:
                results.append({
                    "model": model_name, "role": role_key,
                    "json_valid": False, "json_error": f"http_error:{e}"[:80],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                print(f"FAIL(http_error)")
                continue

            content = resp.get("content", "")
            json_str = extract_json(content)
            valid, err = validate_json(json_str)

            results.append({
                "model": model_name,
                "role": role_key,
                "json_valid": valid,
                "json_error": err,
                "finish_reason": resp.get("finish_reason", ""),
                "content_preview": content[:200],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            status = "OK" if valid else f"FAIL({err[:30]})"
            print(status)

        # Step 4: Kill server
        kill_server()
        time.sleep(1)

        # Save incrementally after each model
        out_path = OUT_DIR / "gguf_sequential_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # ═══════════════════════════ Summary ═══════════════════════════
    print(f"\n{'='*60}")
    print("SUMMARY: GGUF Sequential D3 Role Profiling")
    print(f"{'='*60}")

    per_model: dict = defaultdict(lambda: defaultdict(lambda: {"total": 0, "valid": 0}))
    for r in results:
        per_model[r["model"]][r["role"]]["total"] += 1
        if r["json_valid"]:
            per_model[r["model"]][r["role"]]["valid"] += 1

    for model in sorted(per_model.keys()):
        total = sum(v["total"] for v in per_model[model].values())
        valid = sum(v["valid"] for v in per_model[model].values())
        rate = valid / total * 100 if total > 0 else 0
        roles_str = " | ".join(
            f"{r}:{per_model[model][r]['valid']}/{per_model[model][r]['total']}"
            for r in ["PI", "EC", "RC", "LD"]
        )
        print(f"  {model[:50]}: {valid}/{total} ({rate:.0f}%)  [{roles_str}]")

    print(f"\nResults saved to: {out_path}")
    print(f"Total records: {len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
