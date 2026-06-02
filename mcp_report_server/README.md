# MCP Report Generator — Drug BD Assessment Report Server

A Model Context Protocol (MCP) server that generates BD project assessment reports from drug raw data.

## Quick Start

```bash
# 1. Install dependencies
pip install mcp jinja2 requests pyyaml

# 2. Set LLM API key (if using built-in LLM mode)
export LLM_API_KEY="sk-xxx"
export LLM_API_BASE="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o"

# 3. Run the server (stdio mode)
python server.py
```

## Integration with Workflow Platform

The server exposes one tool: **`generate_report`**

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `raw_data` | object | ✅ | Drug raw data (see below) |
| `analysis_data` | object | ❌ | Pre-generated AI analysis (skips LLM call) |
| `llm_config` | object | ❌ | LLM config (needed if no `analysis_data`) |
| `output_mode` | string | ❌ | `html` / `json` / `both` (default: `html`) |

### raw_data Structure

```json
{
  "drug_core": {
    "drug_id": "SOTORASIB_001",
    "generic_name_cn": "索托拉西布",
    "generic_name_en": "Sotorasib",
    "brand_name": "LUMAKRAS",
    "brand_name_cn": "洛满舒",
    "drug_type": "小分子化药",
    "drug_subtype": "KRAS G12C 抑制剂",
    "originator": "Amgen",
    "originator_full": "Amgen, Inc.",
    "target_name": "KRAS G12C",
    "dosage_form": "片剂",
    "route_admin": "口服",
    "tags": ["KRAS G12C", "First-in-Class", "FDA 获批"],
    "mechanism_desc": "...",
    "mechanism_source": "...",
    "highest_phase": "批准上市",
    "highest_phase_detail": "FDA/EMA/PMDA"
  },
  "ta_coverage": [
    {"indication": "非小细胞肺癌", "indication_type": "肿瘤", "mapped_core_ta": "实体瘤", "mapping_basis": "呼吸系统肿瘤"}
  ],
  "transactions": [
    {"txn_name": "...", "txn_date": "2021-08-19", "transferor": "...", "transferee": "...", "deal_type": "License-in", "rights_region": "中国", "status": "进行中"}
  ],
  "geo_coverage": [
    {"region": "美国", "regulatory_status": "批准上市", "latest_event_date": "2021-05-28", "event_description": "FDA 批准"}
  ],
  "rd_status": [
    {"indication": "NSCLC", "country_region": "美国", "phase": "批准上市", "phase_sort": 7, "organization": "Amgen", "event_date": "2021-05-28"}
  ],
  "clinical_data": [
    {"indication": "NSCLC 2L/3L", "trial_id": "CodeBreaK 200", "endpoint": "mOS", "drug_value": "10.2月", "comparator": "多西他赛", "comparator_value": "6.9月"}
  ],
  "target_related": [
    {"related_drug_name": "Adagrasib", "developer": "Mirati", "highest_phase": "批准上市"}
  ],
  "criteria": [
    {"dimension": "ta", "criteria_text": "核心TA：实体瘤..."},
    {"dimension": "deal_type", "criteria_text": "首选：License-in..."},
    {"dimension": "geo", "criteria_text": "核心区域：中国 + 亚太 + 欧洲"},
    {"dimension": "rd_phase", "criteria_text": "创新药：Phase 2 后期至已上市"}
  ]
}
```

### Usage Modes

**Mode A — Workflow generates analysis (recommended):**
Use your workflow platform's LLM node to generate `analysis_data`, then pass it to this MCP tool for HTML rendering only. No API key needed.

```
[Workflow LLM node] → analysis_data JSON → MCP generate_report(raw_data, analysis_data) → HTML
```

**Mode B — Server generates analysis:**
Pass raw data + LLM config. Server calls LLM internally.

```
MCP generate_report(raw_data, llm_config={api_key: "..."}) → HTML + analysis_json
```

## Configuration

The server checks these environment variables:
- `LLM_API_KEY` — OpenAI-compatible API key
- `LLM_API_BASE` — API endpoint (default: https://api.openai.com/v1)
- `LLM_MODEL` — Model name (default: gpt-4o)
