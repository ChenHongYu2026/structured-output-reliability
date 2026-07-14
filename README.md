# Empirical Findings for Reliable Structured Output in Multi-Model Agent Pipelines

**Hongyu Chen** · AgentSystem Research Lab

[![PDF](https://img.shields.io/badge/PDF-Download-blue)](https://github.com/ChenHongYu2026/structured-output-reliability/raw/main/Empirical%20Findings%20for%20Reliable%20Structured%20Output.pdf)
[![LaTeX](https://img.shields.io/badge/LaTeX-Source-green)](https://github.com/ChenHongYu2026/structured-output-reliability/blob/main/Empirical%20Findings%20for%20Reliable%20Structured%20Output.tex)
[![License](https://img.shields.io/badge/License-Research-lightgrey)]()

---

## TL;DR

LLM agent frameworks assume bad JSON means bad reasoning. **They're wrong.** The bottleneck is *formatting*, not reasoning.

We prove this with a **2×2 factorial experiment** (n=5 per cell, 120 API calls):

|  | Rich Prompt + 800 tokens | Rich Prompt + 30 tokens | Minimal Prompt + 800 tokens | Minimal Prompt + 30 tokens |
|---|---|---|---|---|
| DeepSeek D1–D9 | **100%** | 0% | **100%** | 0% |
| MiniMax D1 | **100%** | 0% | **100%** | 0% |
| MiniMax D9 (EC) | **20%** | 0% | **20%** | 0% |

**`max_tokens` determines everything. Prompt richness does nothing.** And the fix is simple: delegate formatting to a capable model.

| Pipeline | EC Compliance |
|---|---|
| MiniMax → MiniMax (self) | **20%** |
| MiniMax → DeepSeek (delegated) | **100%** |

---

## The Formatting Bottleneck Hypothesis

> Governance reliability in LLM-based agent systems is bounded not by reasoning capability, but by structured output compliance.

**Falsifiable predictions** (all confirmed):

1. Models detect violations at ~100% accuracy when JSON is valid — the failure is in *producing* valid JSON
2. Token budget matters more than prompt quality
3. A strong reasoner + strong formatter outperforms either alone

---

## Key Findings

### 1. Token Budget Is the Dominant Factor

| Property | DeepSeek (PS=0.0) | MiniMax (PS=1.0) |
|---|---|---|
| Dominant factor | `max_tokens` (main effect) | `max_tokens` (main effect) |
| Prompt effect at high tokens | None (both 100%) | None (both 100%) |
| Prompt effect at low tokens | Present (minimal > rich) | None (B=D=0%) |
| Fix | Provision 8000+ tokens | Provision 8000+ tokens |

All low-token failures show `finish_reason=length` — reasoning tokens consume the entire budget before any visible JSON is generated.

### 2. Model Delegation Eliminates Formatting Failures

MiniMax can *reason* perfectly (free-text analysis succeeds 10/10) but cannot *format* reliably (1/5 on EC). Assigning reasoning to MiniMax and formatting to DeepSeek achieves **100% compliance** — an 80 percentage-point improvement.

### 3. SSC Is Decoupled from Model Scale

7 open-source models tested via local inference (llama-server, Vulkan GPU):

| Model | Size | PI | EC | RC | LD |
|---|---|---|---|---|---|
| **HY-MT2-7B** | **7B** | ✅ | ✅ | ✅ | ✅ |
| GPT-OSS-20B | 20B | 0/1 | 0/1* | 0/1 | ✅ |
| Nemotron-30B | 30B | 0/1 | 0/1* | 0/1 | ✅ |
| Gemma-4-26B | 26B | 0/1 | 0/1 | 0/1 | 0/1 |
| Qwen3.6-27B | 27B | 0/1 | 0/1 | 0/1 | 0/1 |

*\*Partial JSON truncated at token limit*

**A 7B model outperformed models 3–4× its size.** Structured-output capability is not a function of parameter count.

### 4. Governance Architectures Are Attack Surfaces

The retry mechanism amplifies adversarial inputs by **25–29×** — a 20-token attack generates ~500 tokens of governance overhead. Attacks succeed by inducing *format collapse*, not reasoning error.

### 5. Cross-Model Pipelines Have a Two-Step Ceiling

Three-step chains fail in 90% of cases. Two-step pipelines achieve 93–100% reliability when the formatting endpoint has adequate SSC tolerance.

---

## Statistical Rigor

We explicitly correct statistical errors common in small-sample LLM studies:

| Issue | Correction |
|---|---|
| Wilcoxon p=0.0039 at n=5 | **p<0.0625** (minimum achievable exact bound) |
| No multiple comparison correction | Bonferroni α=0.05/25=**0.002** |
| Cohen's d without SE | SE ≈ **±0.7** at n=5 per group |

Only binomial exact tests survive Bonferroni. Findings converge across independent experimental lines beyond any single p-value.

---

## Repository Structure
