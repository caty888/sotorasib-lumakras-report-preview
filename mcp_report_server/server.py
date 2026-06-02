from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import (
    Tool, TextContent, ImageContent, EmbeddedResource,
    CallToolResult
)
from pydantic import BaseModel
from typing import Optional, Any
import json
import os
import sys
import asyncio
import logging
import requests
from datetime import datetime
import jinja2

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp-report-server")

server = Server("drug-report-generator")

# ─── Tool: generate_report ─────────────────────────────────────────

GENERATE_REPORT_TOOL = Tool(
    name="generate_report",
    description="从药物原始数据生成BD项目评估报告HTML。支持两种模式：1) 提供raw_data + analysis_data（跳过LLM调用，由工作流平台提前生成分析结论）；2) 仅提供raw_data（服务器内部调用LLM生成分析结论）",
    inputSchema={
        "type": "object",
        "properties": {
            "raw_data": {
                "type": "object",
                "description": "药物原始数据包，包含所有 raw_* 表数据",
                "properties": {
                    "drug_core": {"type": "object", "description": "raw_drug_core 表数据"},
                    "ta_coverage": {"type": "array", "items": {"type": "object"}, "description": "raw_ta_coverage 表数据列表"},
                    "transactions": {"type": "array", "items": {"type": "object"}, "description": "raw_transactions 表数据列表"},
                    "geo_coverage": {"type": "array", "items": {"type": "object"}, "description": "raw_geo_coverage 表数据列表"},
                    "rd_status": {"type": "array", "items": {"type": "object"}, "description": "raw_rd_status 表数据列表"},
                    "clinical_data": {"type": "array", "items": {"type": "object"}, "description": "raw_clinical_data 表数据列表"},
                    "target_related": {"type": "array", "items": {"type": "object"}, "description": "raw_target_related 表数据列表"},
                    "criteria": {"type": "array", "items": {"type": "object"}, "description": "screening_criteria 表数据列表"}
                },
                "required": ["drug_core"]
            },
            "analysis_data": {
                "type": "object",
                "description": "（可选）AI分析结论数据。提供此参数时跳过LLM调用，直接渲染。结构包含 verdict, green_lane, target_summary, conclusion",
                "properties": {
                    "verdict": {"type": "object", "description": "各维度审查结论"},
                    "green_lane": {"type": "object", "description": "机会性绿色通道分析"},
                    "target_summary": {"type": "object", "description": "靶点综合分析"},
                    "conclusion": {"type": "object", "description": "最终研判结论"}
                }
            },
            "llm_config": {
                "type": "object",
                "description": "（可选）LLM调用配置，仅当不提供analysis_data时需要",
                "properties": {
                    "api_key": {"type": "string", "description": "API密钥（默认取环境变量 LLM_API_KEY）"},
                    "api_base": {"type": "string", "description": "API端点（默认取环境变量 LLM_API_BASE）"},
                    "model": {"type": "string", "description": "模型名（默认取环境变量 LLM_MODEL 或 gpt-4o）"},
                    "temperature": {"type": "number", "description": "温度参数，默认0.3"}
                }
            },
            "output_mode": {
                "type": "string",
                "enum": ["html", "json", "both"],
                "description": "输出模式：html=仅返回HTML, json=仅返回结构化JSON, both=两者都返回",
                "default": "html"
            }
        },
        "required": ["raw_data"]
    }
)

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [GENERATE_REPORT_TOOL]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
    if name != "generate_report":
        raise ValueError(f"Unknown tool: {name}")
    
    try:
        raw_data = arguments.get("raw_data", {})
        analysis_data = arguments.get("analysis_data")
        llm_config = arguments.get("llm_config", {})
        output_mode = arguments.get("output_mode", "html")
        
        # 数据校验
        if "drug_core" not in raw_data:
            raise ValueError("raw_data 必须包含 drug_core")
        
        # Step 1: 如果没有提供analysis_data，调用LLM生成
        if not analysis_data:
            logger.info("No analysis_data provided, calling LLM...")
            analysis_data = call_llm_for_analysis(raw_data, llm_config)
            logger.info(f"LLM analysis complete. Verdict: {analysis_data.get('conclusion', {}).get('recommended_action', 'N/A')}")
        else:
            logger.info("Using provided analysis_data")
        
        # Step 2: 渲染HTML
        logger.info("Rendering HTML report...")
        html_report = render_report_html(raw_data, analysis_data)
        
        # Step 3: 准备结果
        drug_name = raw_data.get('drug_core', {}).get('generic_name_cn', '未知药物')
        
        result = {
            "html": html_report,
            "report_id": analysis_data.get('conclusion', {}).get('report_id', 
                f"DR-{datetime.now().strftime('%Y%m%d')}-{raw_data.get('drug_core', {}).get('drug_id', drug_name)[:8].upper()}"),
            "drug_name": drug_name,
            "summary": {
                "verdict": analysis_data.get('conclusion', {}).get('verdict_banner', ''),
                "rating": analysis_data.get('conclusion', {}).get('rating_stars', 0),
                "recommended_action": analysis_data.get('conclusion', {}).get('recommended_action', ''),
                "passed_dimensions": analysis_data.get('conclusion', {}).get('passed_dimensions', [])
            }
        }
        
        if output_mode == "html":
            return [TextContent(type="text", text=json.dumps({"html": html_report, "summary": result["summary"]}, ensure_ascii=False))]
        elif output_mode == "json":
            return [TextContent(type="text", text=json.dumps(analysis_data, ensure_ascii=False, indent=2))]
        else:  # both
            result["analysis_json"] = analysis_data
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            
    except Exception as e:
        logger.error(f"Error in generate_report: {str(e)}", exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


# ─── LLM Integration ────────────────────────────────────────────────

def call_llm_for_analysis(raw_data: dict, llm_config: dict) -> dict:
    """调用LLM生成AI分析结论"""
    
    api_key = llm_config.get('api_key') or os.environ.get('LLM_API_KEY', '')
    api_base = llm_config.get('api_base') or os.environ.get('LLM_API_BASE', 'https://api.openai.com/v1')
    model = llm_config.get('model') or os.environ.get('LLM_MODEL', 'gpt-4o')
    temperature = llm_config.get('temperature', 0.3)
    
    if not api_key:
        raise ValueError("LLM API key not configured. Set LLM_API_KEY env var or pass via llm_config.api_key")
    
    prompt = build_analysis_prompt(raw_data)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个医药BD项目分析师。请严格按JSON格式输出分析结论。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }
    
    try:
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        content = data['choices'][0]['message']['content']
        
        # 解析JSON
        result = json.loads(content)
        
        # 添加report_id
        drug = raw_data.get('drug_core', {})
        if 'conclusion' in result and 'report_id' not in result['conclusion']:
            result['conclusion']['report_id'] = (
                f"DR-{datetime.now().strftime('%Y%m%d')}-{drug.get('drug_id', drug.get('generic_name_en', 'XXX'))[:8].upper()}"
            )
        
        return result
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"LLM API call failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM output as JSON: {str(e)[:200]}")
    except KeyError as e:
        raise RuntimeError(f"Unexpected LLM API response structure: {str(e)}")


# ─── Prompt Builder ─────────────────────────────────────────────────

def build_analysis_prompt(raw_data: dict) -> str:
    """组装分析Prompt"""
    
    drug = raw_data.get('drug_core', {})
    ta_list = raw_data.get('ta_coverage', [])
    transactions = raw_data.get('transactions', [])
    geo_list = raw_data.get('geo_coverage', [])
    rd_list = raw_data.get('rd_status', [])
    clinical = raw_data.get('clinical_data', [])
    target_related = raw_data.get('target_related', [])
    criteria = raw_data.get('criteria', [])
    
    # 阶段排序映射
    phase_sort_map = {
        "药物发现": 1, "临床前": 2, "临床1期": 3, "临床2期": 4,
        "临床3期": 5, "NDA": 6, "批准上市": 7
    }
    max_sort = max((phase_sort_map.get(r.get('phase', ''), 0) for r in rd_list), default=0)
    phase_rev = {v: k for k, v in phase_sort_map.items()}
    highest_phase = phase_rev.get(max_sort, '未知')
    
    # 构建各部分文本
    ta_text = "\n".join(f"- {t.get('indication','')} → {t.get('mapped_core_ta','')}" for t in ta_list) or "无"
    txn_text = "\n".join(
        f"- {t.get('txn_name','')} ({t.get('txn_date','')}), {t.get('deal_type','')}, "
        f"转让方={t.get('transferor','')}, 受让方={t.get('transferee','')}, "
        f"区域={t.get('rights_region','')}, 状态={t.get('status','')}"
        for t in transactions
    ) or "无"
    geo_text = "\n".join(
        f"- {g.get('region','')}: {g.get('regulatory_status','')} ({g.get('latest_event_date','')}) - {g.get('event_description','')}"
        for g in geo_list
    ) or "无"
    rd_text = "\n".join(
        f"- {r.get('indication','')} / {r.get('country_region','')}: {r.get('phase','')} ({r.get('event_date','')})"
        for r in rd_list
    ) or "无"
    clin_text = "\n".join(
        f"- {c.get('indication','')} {c.get('trial_id','')}: {c.get('endpoint','')}={c.get('drug_value','')} "
        f"(对照={c.get('comparator','')}:{c.get('comparator_value','')}, n={c.get('n_patients','')})"
        for c in clinical
    ) or "无"
    target_text = "\n".join(
        f"- {t.get('related_drug_name','')} ({t.get('developer','')}) - {t.get('highest_phase','')}"
        for t in target_related
    ) or "无"
    criteria_text = "\n".join(
        f"[{c.get('dimension','')}] {c.get('criteria_text','')}" for c in criteria
    ) or "无"
    
    return f"""你是一个医药BD项目分析师。请基于以下原始数据，对药物 **{drug.get('generic_name_cn','')}({drug.get('generic_name_en','')})** 进行双轨制敏捷筛选体系分析。

请严格按照以下JSON格式输出分析结论。

## 药物基础信息
- 通用名：{drug.get('generic_name_cn','')} / {drug.get('generic_name_en','')}
- 商品名：{drug.get('brand_name','')}
- 药物类型：{drug.get('drug_type','')}
- 亚类：{drug.get('drug_subtype','')}
- 原研机构：{drug.get('originator','')}
- 靶点：{drug.get('target_name','')}
- 最高研发阶段：{highest_phase}
- 剂型：{drug.get('dosage_form','')}
- 给药途径：{drug.get('route_admin','')}

## 治疗领域覆盖
{ta_text}

## 交易信息
{txn_text}

## 地域权益
{geo_text}

## 研发状态
{rd_text}

## 临床数据
{clin_text}

## 靶点关联药品
{target_text}

## 筛选规则
{criteria_text}

---

现在请你按以下维度逐一分析，输出JSON：

### JSON输出格式：
```json
{{
  "verdict": {{
    "ta": {{
      "pass": true,
      "reasoning": "通过理由",
      "supporting_evidence": ["引用的原始数据摘要"]
    }},
    "deal_type": {{
      "pass": true,
      "reasoning": "通过理由",
      "supporting_evidence": []
    }},
    "geo": {{
      "pass": true,
      "reasoning": "通过理由",
      "supporting_evidence": []
    }},
    "rd_phase": {{
      "pass": true,
      "reasoning": "通过理由",
      "supporting_evidence": []
    }},
    "standard_track": {{
      "label": "标准化快速筛选 — 完全符合审查标准",
      "detail": "总结性描述"
    }},
    "green_lane": {{
      "label": "机会性绿色通道 — 具备触发潜力",
      "detail": "总结性描述"
    }}
  }},
  "green_lane": {{
    "data_disruptiveness": {{
      "triggered": true,
      "summary": "临床数据总结",
      "clinical_refs": ["引用的临床数据"]
    }},
    "mechanism_novelty": {{
      "triggered": true,
      "summary": "机制首创性总结",
      "count": 203
    }},
    "commercial_scarcity": {{
      "triggered": true,
      "summary": "商业稀缺性总结",
      "dosage_form": "片剂",
      "route": "口服"
    }}
  }},
  "target_summary": {{
    "associated_drug_count": 203,
    "highest_competitive_drug": "竞品名",
    "competition_landscape": "竞争格局描述",
    "is_first_in_class": true
  }},
  "conclusion": {{
    "verdict_banner": "✓ 建议推进 License-in 中国权益 — 总结性描述",
    "rating_stars": 5,
    "rating_label": "重点跟进项目",
    "recommended_action": "License-in",
    "passed_dimensions": ["治疗领域 ✓", "合作模式 ✓", "地域权益 ✓", "研发阶段 ✓"],
    "risks": [{{"risk": "风险描述1"}}, {{"risk": "风险描述2"}}, {{"risk": "风险描述3"}}],
    "action_plan": [
      {{"icon": "🤝", "title": "建议1标题", "desc": "建议1描述"}},
      {{"icon": "🔬", "title": "建议2标题", "desc": "建议2描述"}},
      {{"icon": "🌏", "title": "建议3标题", "desc": "建议3描述"}}
    ],
    "bottom_metrics": [
      {{"label": "推荐动作", "value": "License-in"}},
      {{"label": "中国获批预期", "value": "3-4 年"}},
      {{"label": "推荐合作模式", "value": "License-in / Distribution"}},
      {{"label": "报告编号", "value": "DR-20260528-001"}}
    ],
    "report_id": "DR-日期-药物ID"
  }}
}}
```

注意：
1. reasoning 字段写中文，说明判断依据
2. supporting_evidence 引用具体的数据来源
3. risks 至少3条
4. action_plan 至少3条
5. report_id 请生成为 DR-YYYYMMDD-DRUGID 格式
6. rating_stars 根据综合评估打分(1~5)"""


# ─── HTML Template ──────────────────────────────────────────────────

TEMPLATE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ drug.generic_name_cn }}（{{ drug.generic_name_en }}）BD项目评估报告</title>
<link href="https://fonts.googleapis.cn/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--teal:#3547a6;--teal-2:#6577d8;--teal-dark:#1f2a63;--teal-light:#edf0ff;--teal-muted:#f6f7ff;--gold:#d16d5f;--gold-light:#fff0ed;--gold-muted:#fff7f5;--gold-dark:#a94d43;--ink:#171927;--ink-2:#31354a;--ink-3:#62677d;--ink-4:#9297a9;--rule:#e0e4f0;--rule-2:#c5ccdf;--bg:#f6f7fb;--bg-2:#eceff8;--bg-card:#ffffff;--green:#26725f;--green-light:#e7f3ef;--purple:#6a4ca6;--purple-light:#f0ebfb}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:16px;scroll-behavior:smooth}
body{font-family:'Noto Sans SC',sans-serif;background:var(--bg);color:var(--ink);line-height:1.8;min-height:100vh}
.cover{background:linear-gradient(145deg,#1b245c 0%,#3547a6 52%,#7d8bea 100%);color:#fff;padding:80px 60px 60px;position:relative;overflow:hidden}
.cover::before{content:'';position:absolute;top:0;right:0;bottom:0;left:0;background:radial-gradient(ellipse 640px 420px at 78% 18%,rgba(209,109,95,0.16) 0%,transparent 62%),radial-gradient(ellipse 300px 300px at 20% 80%,rgba(255,255,255,0.03) 0%,transparent 50%);pointer-events:none}
.cover-pattern{position:absolute;top:-60px;right:-60px;width:500px;height:500px;border:1px solid rgba(255,255,255,0.10);border-radius:50%;pointer-events:none}
.cover-pattern::after{content:'';position:absolute;top:60px;right:60px;width:380px;height:380px;border:1px solid rgba(255,255,255,0.08);border-radius:50%}
.cover-deco-line{position:absolute;bottom:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent 5%,#ffb4aa 28%,#d16d5f 72%,transparent 95%);opacity:0.55}
.cover-label{position:relative;z-index:1;font-size:11px;letter-spacing:0.25em;text-transform:uppercase;color:rgba(255,255,255,0.45);margin-bottom:28px;font-family:'JetBrains Mono',monospace}
.cover-title{position:relative;z-index:1;font-family:'Noto Serif SC',serif;font-size:42px;font-weight:700;line-height:1.25;margin-bottom:8px;letter-spacing:-0.01em}
.cover-title-sub{font-size:18px;color:rgba(255,255,255,0.6);margin-bottom:36px;font-weight:300;position:relative;z-index:1}
.cover-tags{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:44px;position:relative;z-index:1}
.cover-tag{font-size:11px;background:rgba(255,255,255,0.10);border:1px solid rgba(255,255,255,0.20);color:rgba(255,255,255,0.8);padding:4px 14px;border-radius:3px;letter-spacing:0.05em;font-weight:400}
.cover-meta{display:flex;gap:48px;border-top:1px solid rgba(255,255,255,0.12);padding-top:24px;position:relative;z-index:1;flex-wrap:wrap}
.cover-meta-item{font-size:13px}
.cover-meta-label{color:rgba(255,255,255,0.35);margin-bottom:4px;font-size:10px;letter-spacing:0.15em;text-transform:uppercase}
.cover-meta-value{color:rgba(255,255,255,0.85);font-weight:500}
.toc{background:rgba(255,255,255,0.95);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-bottom:1px solid var(--rule);padding:0 60px;position:sticky;top:0;z-index:100}
.toc-inner{display:flex;gap:0;overflow-x:auto;max-width:1040px;margin:0 auto;padding:14px 0}
.toc-inner::-webkit-scrollbar{height:0}
.toc-item{font-size:12px;color:var(--ink-3);text-decoration:none;padding:6px 18px;border-radius:4px;white-space:nowrap;transition:all 0.2s ease;font-weight:500;letter-spacing:0.02em;border:1px solid transparent;flex-shrink:0}
.toc-item:hover{background:var(--teal-light);color:var(--teal);border-color:rgba(53,71,166,0.15)}
.main{max-width:1040px;margin:0 auto;padding:56px 40px}
@media(max-width:1024px){.main{padding:40px 32px}}
.section{margin-bottom:52px}
.section-header{display:flex;align-items:baseline;gap:16px;margin-bottom:24px;padding-bottom:12px;border-bottom:2px solid var(--teal)}
.section-num{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--teal);background:var(--teal-light);padding:2px 10px;border-radius:3px;font-weight:500}
.section-title{font-family:'Noto Serif SC',serif;font-size:24px;font-weight:700;color:var(--teal-dark);letter-spacing:-0.01em}
.section-desc{font-size:13px;color:var(--ink-3);margin:-18px 0 20px;line-height:1.6}
p{font-size:14px;color:var(--ink-2);line-height:1.85;margin-bottom:14px}
.keydata{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin:24px 0;align-items:stretch}
.keydata-card{min-width:0;background:var(--bg-card);border:1px solid var(--rule);border-radius:8px;padding:18px 16px;transition:box-shadow 0.2s ease;height:100%}
.keydata-card:hover{box-shadow:0 8px 24px rgba(53,71,166,0.10)}
.keydata-label{font-size:10px;color:var(--ink-4);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;font-weight:500}
.keydata-value{font-size:20px;font-weight:700;color:var(--teal);font-family:'Noto Serif SC',serif;line-height:1.2}
.keydata-sub{font-size:11px;color:var(--ink-4);margin-top:4px}
.screen-criteria{background:var(--teal-light);border-radius:8px;padding:14px 18px;margin-bottom:10px;font-size:13px;color:var(--ink-2);line-height:1.8;border-left:3px solid var(--teal)}
.screen-criteria strong{color:var(--teal)}
.screen-result{display:flex;align-items:flex-start;gap:12px;padding:14px 18px;margin-bottom:14px;background:var(--bg-card);border-radius:8px;border:1px solid var(--green-light)}
.screen-icon{flex-shrink:0;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;background:var(--green-light);color:var(--green)}
.screen-text{font-size:13px;color:var(--ink-2);line-height:1.7}
.screen-text strong{color:var(--green)}
.screen-data{background:var(--bg-card);border:1px solid var(--rule);border-radius:8px;padding:14px 18px;margin:8px 0 10px}
.screen-data-title{font-size:12px;font-weight:600;color:var(--ink-3);margin-bottom:6px;letter-spacing:0.05em}
.screen-data p{font-size:13px;margin-bottom:0;line-height:1.8}
.screen-data p + p{margin-top:6px}
.table-wrap{overflow-x:auto;margin:16px 0;border-radius:8px;border:1px solid var(--rule)}
table{width:100%;border-collapse:collapse;font-size:13px;background:var(--bg-card)}
thead tr{background:var(--teal);color:#fff}
thead th{padding:10px 14px;text-align:left;font-weight:600;font-size:12px;letter-spacing:0.04em;white-space:nowrap}
tbody tr{border-bottom:1px solid var(--rule);transition:background 0.15s ease}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:var(--teal-muted)}
tbody td{padding:9px 14px;color:var(--ink-2);line-height:1.5;vertical-align:top}
.td-bold{font-weight:600;color:var(--ink)}
.badge{display:inline-block;font-size:10px;font-weight:600;padding:2px 10px;border-radius:3px;letter-spacing:0.05em;white-space:nowrap}
.badge-approved{background:var(--green-light);color:var(--green)}
.badge-clinical{background:var(--purple-light);color:var(--purple)}
.badge-other{background:var(--gold-light);color:var(--gold-dark)}
.target-card{background:var(--bg-card);border:1px solid var(--rule);border-radius:8px;padding:18px 20px;margin:16px 0}
.target-header{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.target-icon{width:28px;height:28px;border-radius:6px;background:var(--teal-light);display:flex;align-items:center;justify-content:center;font-size:14px;color:var(--teal)}
.target-name{font-size:15px;font-weight:700;color:var(--teal-dark)}
.target-divider{height:1px;background:var(--rule);margin:10px 0}
.target-row{display:flex;justify-content:space-between;padding:5px 0;font-size:13px;border-bottom:1px solid var(--rule)}
.target-row:last-child{border-bottom:none}
.target-label{color:var(--ink-3)}
.target-value{font-weight:500;color:var(--ink)}
.t2-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}
@media(max-width:768px){.t2-grid{grid-template-columns:1fr}}
.t2-card{background:var(--bg-card);border:1px solid var(--rule);border-radius:8px;padding:18px;transition:box-shadow 0.2s ease;height:100%}
.t2-card:hover{box-shadow:0 4px 16px rgba(53,71,166,0.08)}
.t2-title{font-size:14px;font-weight:700;color:var(--teal);margin-bottom:4px}
.t2-source{font-size:11px;color:var(--ink-4);margin-bottom:10px}
.t2-hl{background:var(--gold-muted);border-left:3px solid var(--gold);padding:10px 14px;border-radius:0 6px 6px 0;margin-bottom:10px}
.t2-hl-title{font-size:12px;font-weight:700;color:var(--gold-dark);margin-bottom:4px}
.t2-hl p{font-size:12px;color:var(--ink-2);margin:0;line-height:1.7}
.t2-hl p + p{margin-top:4px}
.t2-card ul{list-style:none;padding:0}
.t2-card li{font-size:13px;color:var(--ink-2);padding:4px 0 4px 14px;position:relative;line-height:1.6;clear:both}
.t2-card li::before{content:'•';position:absolute;left:0;color:var(--gold)}
.dose-row{display:flex;gap:12px;margin:10px 0}
.dose-item{flex:1;border:1px solid var(--rule);border-radius:6px;padding:12px;text-align:center;background:var(--bg-card)}
.dose-label{font-size:10px;color:var(--ink-4);letter-spacing:0.1em;margin-bottom:4px}
.dose-value{font-size:17px;font-weight:700;color:var(--teal)}
.verdict-banner{background:linear-gradient(135deg,var(--teal-dark),var(--teal));color:#fff;border-radius:10px;padding:20px 28px;margin:24px 0;font-size:14px;line-height:1.7}
.verdict-banner strong{color:#ffd4aa}
.footer{background:var(--ink);color:rgba(255,255,255,0.35);text-align:center;padding:28px 24px;font-size:11px;letter-spacing:0.05em;line-height:1.8}
.footer strong{color:rgba(255,255,255,0.5)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){
.cover{padding:48px 24px 40px}
.cover-title{font-size:28px}
.cover-meta{gap:20px}
.toc{padding:0 16px}
.main{padding:32px 20px}
.keydata{grid-template-columns:repeat(3,minmax(0,1fr))}
}
@media(max-width:600px){
.cover{padding:32px 16px 28px}
.cover-title{font-size:22px}
.cover-meta{gap:14px;flex-direction:column}
.toc{padding:0 8px}
.toc-inner{padding:10px 0}
.toc-item{font-size:11px;padding:5px 12px}
.main{padding:24px 14px}
.section-title{font-size:18px}
.keydata{grid-template-columns:repeat(2,minmax(0,1fr))}
.t2-grid{grid-template-columns:1fr}
}
@media print{.toc{position:static;backdrop-filter:none;background:var(--bg-2)}.cover{page-break-after:always}.section{page-break-inside:avoid}}
</style>
</head>
<body>

<div class="cover">
  <div class="cover-pattern"></div>
  <div class="cover-deco-line"></div>
  <div class="cover-label">药物项目检索报告 · 内部参考文件 · {{ report.report_id }}</div>
  <div class="cover-title">{{ drug.generic_name_cn }}（{{ drug.generic_name_en }}）<br>{{ drug.brand_name }} 项目评估报告</div>
  <div class="cover-title-sub">{{ drug.drug_subtype }} · {{ drug.tagline }}</div>
  <div class="cover-tags">
    {% for tag in drug.tags_list %}
    <span class="cover-tag">{{ tag }}</span>
    {% endfor %}
  </div>
  <div class="cover-meta">
    <div class="cover-meta-item"><div class="cover-meta-label">报告生成时间</div><div class="cover-meta-value">{{ report.generated_at }}</div></div>
    <div class="cover-meta-item"><div class="cover-meta-label">数据来源</div><div class="cover-meta-value">{{ report.data_source_label }}</div></div>
    <div class="cover-meta-item"><div class="cover-meta-label">原研机构</div><div class="cover-meta-value">{{ drug.originator_full }}</div></div>
  </div>
</div>

<nav class="toc">
  <div class="toc-inner">
    <a href="#s0" class="toc-item">基础信息</a>
    <a href="#s1" class="toc-item">治疗领域 (TA) 扩展</a>
    <a href="#s2" class="toc-item">合作模式 (Deal Type)</a>
    <a href="#s3" class="toc-item">地域权益 (Geography)</a>
    <a href="#s4" class="toc-item">研发阶段</a>
    <a href="#s5" class="toc-item">机会性绿色通道</a>
    <a href="#s6" class="toc-item">研判结论</a>
  </div>
</nav>

<div class="main">

<section class="section" id="s0">
  <div class="section-header"><span class="section-num">00</span><h2 class="section-title">基础信息</h2></div>
  <div class="keydata">
    <div class="keydata-card"><div class="keydata-label">通用名</div><div class="keydata-value">{{ drug.generic_name_cn }}</div><div class="keydata-sub">{{ drug.generic_name_en }}</div></div>
    <div class="keydata-card"><div class="keydata-label">商品名</div><div class="keydata-value">{{ drug.brand_name }}</div><div class="keydata-sub">{{ drug.brand_name_cn or '' }}</div></div>
    <div class="keydata-card"><div class="keydata-label">药物类型</div><div class="keydata-value">{{ drug.drug_type }}</div><div class="keydata-sub">{{ drug.drug_subtype }}</div></div>
    <div class="keydata-card"><div class="keydata-label">原研机构</div><div class="keydata-value">{{ drug.originator }}</div><div class="keydata-sub">{{ drug.originator_full }}</div></div>
    <div class="keydata-card"><div class="keydata-label">最高研发阶段</div><div class="keydata-value">{{ drug.highest_phase }}</div><div class="keydata-sub">{{ drug.highest_phase_detail }}</div></div>
  </div>

  {% if drug.mechanism_desc %}
  <div class="subsection" style="margin-top:28px;">
    <div class="subsection-title" style="font-size:16px;font-weight:700;color:var(--teal);margin-bottom:14px;display:flex;align-items:center;gap:10px;"><span style="display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:3px;background:var(--gold-muted);color:var(--gold-dark);font-size:11px;flex-shrink:0;">●</span>靶点与作用机制</div>
    <p>{{ drug.mechanism_desc }}</p>
    {% if drug.mechanism_source %}
    <div style="background:var(--teal-light);border-left:3px solid var(--teal);padding:8px 14px;border-radius:0 6px 6px 0;font-size:12px;color:var(--ink-3);">来源: {{ drug.mechanism_source }}</div>
    {% endif %}
  </div>
  {% endif %}

  <div class="subsection" style="margin-top:28px;">
    <div class="subsection-title" style="font-size:16px;font-weight:700;color:var(--teal);margin-bottom:14px;display:flex;align-items:center;gap:10px;"><span style="display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:3px;background:var(--gold-muted);color:var(--gold-dark);font-size:11px;flex-shrink:0;">◆</span>适配判定</div>

    {% if analysis.verdict.standard_track %}
    <div style="display:flex;align-items:flex-start;gap:12px;padding:14px 18px;margin-bottom:10px;background:var(--bg-card);border-radius:8px;border:1px solid var(--green-light);">
      <div style="flex-shrink:0;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;background:var(--green-light);color:var(--green);">✓</div>
      <div style="flex:1;">
        <div style="font-size:14px;font-weight:700;color:var(--ink);margin-bottom:3px;">{{ analysis.verdict.standard_track.label }}</div>
        <div style="font-size:13px;color:var(--ink-2);line-height:1.7;">{{ analysis.verdict.standard_track.detail }}</div>
      </div>
    </div>
    {% endif %}

    {% if analysis.verdict.green_lane %}
    <div style="display:flex;align-items:flex-start;gap:12px;padding:14px 18px;margin-bottom:10px;background:var(--bg-card);border-radius:8px;border:1px solid var(--gold-light);">
      <div style="flex-shrink:0;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;background:var(--gold-light);color:var(--gold-dark);">★</div>
      <div style="flex:1;">
        <div style="font-size:14px;font-weight:700;color:var(--ink);margin-bottom:3px;">{{ analysis.verdict.green_lane.label }}</div>
        <div style="font-size:13px;color:var(--ink-2);line-height:1.7;">{{ analysis.verdict.green_lane.detail }}</div>
      </div>
    </div>
    {% endif %}
  </div>
</section>

<!-- TA -->
<section class="section" id="s1">
  <div class="section-header"><span class="section-num">01</span><h2 class="section-title">治疗领域 (TA) 扩展</h2></div>
  <div class="section-desc">标准化筛选条件：是否落在核心 TA 范围内</div>
  {% for c in criteria %}{% if c.dimension == 'ta' %}
  <div class="screen-criteria"><strong>核心TA</strong>：{{ c.criteria_text }}</div>
  {% endif %}{% endfor %}

  <div class="screen-data">
    <div class="screen-data-title">本品覆盖领域</div>
    {% for ta in ta_coverage %}
    <p><strong>{{ ta.indication }}</strong> → {{ ta.mapping_basis or ta.mapped_core_ta }}</p>
    {% endfor %}
  </div>

  {% if analysis.verdict.ta %}
  <div class="screen-result">
    <div class="screen-icon">{{ '✓' if analysis.verdict.ta.pass else '✗' }}</div>
    <div class="screen-text"><strong>{{ '通过' if analysis.verdict.ta.pass else '不通过' }}</strong> — {{ analysis.verdict.ta.reasoning }}</div>
  </div>
  {% endif %}
</section>

<!-- Deal Type -->
<section class="section" id="s2">
  <div class="section-header"><span class="section-num">02</span><h2 class="section-title">合作模式 (Deal Type)</h2></div>
  <div class="section-desc">标准化筛选条件：合作类型是否符合首选模式</div>
  {% for c in criteria %}{% if c.dimension == 'deal_type' %}
  <div class="screen-criteria">{{ c.criteria_text | replace('\\\\n', '<br>') | safe }}</div>
  {% endif %}{% endfor %}

  {% if transactions %}
  <div class="screen-data">
    <div class="screen-data-title">已有引进先例</div>
    {% for t in transactions %}
    {% if t.rights_region and '中国' in t.rights_region %}
    <p><strong>{{ t.transferee }}（{{ t.txn_date }}）</strong>：{{ t.txn_name }}，属于 <strong>{{ t.deal_type }}</strong> 模式</p>
    {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  {% if analysis.verdict.deal_type %}
  <div class="screen-result">
    <div class="screen-icon">{{ '✓' if analysis.verdict.deal_type.pass else '✗' }}</div>
    <div class="screen-text"><strong>{{ '通过' if analysis.verdict.deal_type.pass else '不通过' }}</strong> — {{ analysis.verdict.deal_type.reasoning }}</div>
  </div>
  {% endif %}
</section>

<!-- Geography -->
<section class="section" id="s3">
  <div class="section-header"><span class="section-num">03</span><h2 class="section-title">地域权益 (Geography)</h2></div>
  <div class="section-desc">标准化筛选条件：是否覆盖核心权益区域</div>
  {% for c in criteria %}{% if c.dimension == 'geo' %}
  <div class="screen-criteria"><strong>核心区域</strong>：{{ c.criteria_text }}</div>
  {% endif %}{% endfor %}

  <div class="screen-data">
    <div class="screen-data-title">权益覆盖情况</div>
    {% for g in geo_coverage %}
    <p><strong>{{ g.region }}</strong>：{{ g.event_description or g.regulatory_status }}{% if g.latest_event_date %}（{{ g.latest_event_date }}）{% endif %}</p>
    {% endfor %}
  </div>

  {% if analysis.verdict.geo %}
  <div class="screen-result">
    <div class="screen-icon">{{ '✓' if analysis.verdict.geo.pass else '✗' }}</div>
    <div class="screen-text"><strong>{{ '通过' if analysis.verdict.geo.pass else '不通过' }}</strong> — {{ analysis.verdict.geo.reasoning }}</div>
  </div>
  {% endif %}

  {% if transactions %}
  <div class="section-header" style="margin-top:28px;border-bottom-color:var(--rule);padding-bottom:8px;margin-bottom:14px;"><span style="font-size:12px;color:var(--ink-3);font-weight:500;">核心交易信息</span></div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th style="width:32px;">#</th>
          <th>交易名称</th>
          <th>时间</th>
          <th>转让方</th>
          <th>受让方</th>
          <th>交易类型</th>
          <th>权益地区</th>
          <th>状态</th>
        </tr>
      </thead>
      <tbody>
        {% for t in transactions %}
        <tr{% if t.rights_region and '中国' in t.rights_region %} style="background:var(--teal-muted);"{% endif %}>
          <td class="td-bold" style="text-align:center;">{{ loop.index }}</td>
          <td>{{ t.txn_name }}</td>
          <td>{{ t.txn_date }}</td>
          <td>{{ t.transferor }}</td>
          <td>{{ t.transferee }}</td>
          <td>{{ t.deal_type }}</td>
          <td>{{ t.rights_region }}</td>
          <td><span class="badge badge-approved">{{ t.status or '进行中' }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  <p style="font-size:12px;color:var(--ink-4);margin-top:8px;">以上数据来源：药品商业数据库公开信息。标注浅蓝色背景的行为中国权益相关交易。</p>
  {% endif %}
</section>

<!-- R&D Phase -->
<section class="section" id="s4">
  <div class="section-header"><span class="section-num">04</span><h2 class="section-title">研发阶段</h2></div>
  <div class="section-desc">标准化筛选条件：是否在允许的研发阶段范围内</div>
  {% for c in criteria %}{% if c.dimension == 'rd_phase' %}
  <div class="screen-criteria">{{ c.criteria_text | replace('\\\\n', '<br>') | safe }}</div>
  {% endif %}{% endfor %}

  <div class="screen-data">
    <div class="screen-data-title">研发状态总览</div>
    <p><strong>最高研发阶段</strong>：<span class="badge badge-approved">{{ drug.highest_phase }}</span>{{ drug.highest_phase_detail and ('（' + drug.highest_phase_detail + '）') or '' }}</p>
    <p><strong>靶点</strong>：{{ drug.target_name }}</p>
    {% for g in geo_coverage %}{% if g.region == '中国' %}
    <p><strong>中国推进阶段</strong>：{{ g.regulatory_status }}{% if g.latest_event_date %}（{{ g.latest_event_date }}）{% endif %}</p>
    {% endif %}{% endfor %}
  </div>

  {% if analysis.verdict.rd_phase %}
  <div class="screen-result">
    <div class="screen-icon">{{ '✓' if analysis.verdict.rd_phase.pass else '✗' }}</div>
    <div class="screen-text"><strong>{{ '通过' if analysis.verdict.rd_phase.pass else '不通过' }}</strong> — {{ analysis.verdict.rd_phase.reasoning }}</div>
  </div>
  {% endif %}

  {% if rd_status %}
  <div class="target-card">
    <div class="target-header">
      <div class="target-icon">🎯</div>
      <div class="target-name">{{ drug.target_name }}</div>
    </div>
    <div class="target-divider"></div>
    <div class="target-row"><span class="target-label">靶点名称</span><span class="target-value">{{ drug.target_name }}</span></div>
    <div class="target-row"><span class="target-label">最高研发状态</span><span class="target-value"><span class="badge badge-approved">{{ drug.highest_phase }}</span></span></div>
    {% if analysis.target_summary %}
    <div class="target-row"><span class="target-label">关联药物数量</span><span class="target-value" style="color:var(--teal);font-weight:600;">{{ analysis.target_summary.associated_drug_count }}+</span></div>
    <div class="target-row"><span class="target-label">最高研发药物</span><span class="target-value">Sotorasib（{{ drug.highest_phase }}）</span></div>
    <div class="target-row"><span class="target-label">已上市竞品</span><span class="target-value">{{ analysis.target_summary.highest_competitive_drug }}</span></div>
    {% endif %}
  </div>

  <div class="section-header" style="margin-top:28px;border-bottom-color:var(--rule);padding-bottom:8px;margin-bottom:14px;"><span style="font-size:12px;color:var(--ink-3);font-weight:500;">研发状态明细</span></div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>适应症</th><th>国家/地区</th><th>最高状态</th><th>机构</th><th>日期</th></tr></thead>
      <tbody>
        {% for r in rd_status %}
        <tr>
          <td class="td-bold">{{ r.indication }}</td>
          <td>{{ r.country_region }}</td>
          <td><span class="badge badge-{{ 'approved' if r.phase == '批准上市' else 'clinical' if '临床' in r.phase else 'other' }}">{{ r.phase }}</span></td>
          <td>{{ r.organization }}</td>
          <td>{{ r.event_date }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}
</section>

<!-- Green Lane -->
<section class="section" id="s5">
  <div class="section-header"><span class="section-num">05</span><h2 class="section-title">机会性绿色通道 (Opportunistic Green Lane)</h2></div>
  <div class="section-desc">标准化快速筛选通过后，额外关注以下三项指标</div>

  <div class="t2-grid">
    <!-- 数据颠覆性 -->
    <div class="t2-card">
      <div class="t2-title">✦ 数据颠覆性</div>
      <div class="t2-source">判断依据：临床结果数据</div>
      {% if analysis.green_lane and analysis.green_lane.data_disruptiveness %}
      <div class="t2-hl">
        {% if analysis.green_lane.data_disruptiveness.clinical_refs %}
        {% for ref in analysis.green_lane.data_disruptiveness.clinical_refs %}
        <p>{{ ref }}</p>
        {% endfor %}
        {% endif %}
      </div>
      <p style="font-size:12px;color:var(--ink-3);">{{ analysis.green_lane.data_disruptiveness.summary }}</p>
      {% endif %}
    </div>

    <!-- 机制首创性 -->
    <div class="t2-card">
      <div class="t2-title">◆ 机制首创性</div>
      <div class="t2-source">判断依据：靶点-关联药品数量（<5 满足，>5 展示明细）</div>
      {% if analysis.target_summary %}
      <div class="t2-hl" style="border-left-color:var(--teal);background:var(--teal-muted);">
        <p style="font-size:13px;"><strong>关联药物数量 {{ analysis.target_summary.associated_drug_count }}+{% if analysis.target_summary.associated_drug_count > 5 %}（>5），明细如下{% endif %}</strong></p>
      </div>
      <ul>
        {% if target_related %}
        {% for dr in target_related[:7] %}
        <li><strong>{{ dr.related_drug_name }}</strong> <span class="badge badge-{{ 'approved' if dr.highest_phase == '批准上市' else 'clinical' if '临床' in dr.highest_phase else 'other' }}" style="float:right;">{{ dr.highest_phase }}</span></li>
        {% endfor %}
        {% endif %}
      </ul>
      {% endif %}
    </div>

    <!-- 商业稀缺性 -->
    <div class="t2-card">
      <div class="t2-title">▲ 商业稀缺性</div>
      <div class="t2-source">判断依据：批准-剂型（去重）</div>
      <div class="dose-row">
        <div class="dose-item"><div class="dose-label">剂型</div><div class="dose-value">{{ drug.dosage_form }}</div></div>
        <div class="dose-item"><div class="dose-label">给药途径</div><div class="dose-value">{{ drug.route_admin }}</div></div>
      </div>
      {% if analysis.green_lane and analysis.green_lane.commercial_scarcity %}
      <p style="font-size:12px;color:var(--ink-3);">{{ analysis.green_lane.commercial_scarcity.summary }}</p>
      {% endif %}
    </div>
  </div>
</section>

<!-- Conclusion -->
<section class="section" id="s6">
  <div class="section-header"><span class="section-num">06</span><h2 class="section-title">研判结论</h2></div>

  {% if analysis.conclusion %}
  <div class="verdict-banner">
    <strong>{{ analysis.conclusion.verdict_banner.split(' — ')[0] if ' — ' in analysis.conclusion.verdict_banner else analysis.conclusion.verdict_banner }}</strong> — {{ analysis.conclusion.verdict_banner.split(' — ', 1)[1] if ' — ' in analysis.conclusion.verdict_banner else '' }}
  </div>

  <div class="screen-data">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
      <div>
        <div style="font-size:11px;letter-spacing:0.12em;color:var(--ink-4);margin-bottom:2px;">综合评级</div>
        <div style="font-size:20px;font-weight:700;letter-spacing:2px;color:#d16d5f;">{{ '★' * analysis.conclusion.rating_stars }}{{ '☆' * (5 - analysis.conclusion.rating_stars) }}</div>
        <div style="font-size:13px;color:var(--ink-2);margin-top:2px;">{{ analysis.conclusion.rating_label }}</div>
      </div>
      <div style="background:var(--teal-light);border-radius:6px;padding:8px 16px;text-align:center;">
        <div style="font-size:11px;letter-spacing:0.12em;color:var(--teal);margin-bottom:2px;font-weight:500;">研判结论</div>
        <div style="font-size:15px;font-weight:700;color:var(--teal);">✓ 建议推进 {{ analysis.conclusion.recommended_action }}</div>
      </div>
    </div>
  </div>

  {% if analysis.conclusion.passed_dimensions or analysis.conclusion.risks %}
  <div class="grid2">
    {% if analysis.conclusion.passed_dimensions %}
    <div class="screen-data" style="margin:0;">
      <div style="font-size:14px;font-weight:700;color:var(--teal);margin-bottom:10px;display:flex;align-items:center;gap:6px;">✅ 轨道审查通过项</div>
      <div class="grid2" style="gap:6px;">
        {% for dim in analysis.conclusion.passed_dimensions %}
        <div style="background:var(--green-light);border-radius:4px;padding:5px 10px;font-size:12px;font-weight:500;color:var(--green);text-align:center;">{{ dim }}</div>
        {% endfor %}
      </div>
    </div>
    {% endif %}
    {% if analysis.conclusion.risks %}
    <div class="screen-data" style="margin:0;">
      <div style="font-size:14px;font-weight:700;color:var(--gold-dark);margin-bottom:10px;display:flex;align-items:center;gap:6px;">⚠️ 风险提示</div>
      <div style="font-size:13px;color:var(--ink-2);line-height:1.7;">
        {% for risk in analysis.conclusion.risks %}
        • {{ risk.risk }}<br>
        {% endfor %}
      </div>
    </div>
    {% endif %}
  </div>
  {% endif %}

  {% if analysis.conclusion.action_plan %}
  <div style="font-size:15px;font-weight:700;color:var(--teal);margin:20px 0 12px;">行动建议</div>
  <div class="t2-grid" style="margin:0 0 14px 0;">
    {% for action in analysis.conclusion.action_plan %}
    <div class="t2-card" style="padding:14px 18px;">
      <div style="font-size:22px;margin-bottom:4px;">{{ action.icon }}</div>
      <div style="font-size:14px;font-weight:700;color:var(--teal);margin-bottom:4px;">{{ action.title }}</div>
      <div style="font-size:13px;color:var(--ink-3);line-height:1.6;">{{ action.desc }}</div>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  {% if analysis.conclusion.bottom_metrics %}
  <div class="screen-data" style="margin:0;padding:0;">
    <div class="grid2" style="grid-template-columns:repeat(4,1fr);gap:0;">
      {% for m in analysis.conclusion.bottom_metrics %}
      <div style="padding:12px 8px;{% if not loop.last %}border-right:1px solid var(--rule);{% endif %}text-align:center;">
        <div style="font-size:10px;color:var(--ink-4);letter-spacing:0.1em;margin-bottom:2px;">{{ m.label }}</div>
        <div style="font-size:13px;font-weight:600;color:var(--teal);">{{ m.value }}</div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}
  {% endif %}
</section>

</div>

<div class="footer">
  <strong>{{ drug.generic_name_cn }}（{{ drug.generic_name_en }} / {{ drug.brand_name }}）项目评估报告</strong><br>
  报告生成：{{ report.generated_at }} &nbsp;·&nbsp; {{ report.framework_name }} &nbsp;·&nbsp; 仅供内部参考
</div>

</body>
</html>"""


# ─── HTML Renderer ──────────────────────────────────────────────────

def render_report_html(raw_data: dict, analysis_data: dict) -> str:
    """用Jinja2渲染报告HTML"""
    
    drug_core = raw_data.get('drug_core', {})
    
    # 处理tags
    tags_raw = drug_core.get('tags', [])
    if isinstance(tags_raw, str):
        try:
            tags_list = json.loads(tags_raw)
        except (json.JSONDecodeError, TypeError):
            tags_list = [tags_raw]
    else:
        tags_list = tags_raw or []
    
    # 处理tagline
    tagline = drug_core.get('tagline', f"{drug_core.get('target_name', '')} 抑制剂 · 突破性创新药物")
    
    # 处理highest_phase_detail
    highest_phase_detail = drug_core.get('highest_phase_detail', '')
    
    # 处理报告元信息
    report_id = analysis_data.get('conclusion', {}).get(
        'report_id',
        f"DR-{datetime.now().strftime('%Y%m%d')}-{drug_core.get('drug_id', 'XXXX')[:8].upper()}"
    )
    
    template_data = {
        "report": {
            "report_id": report_id,
            "generated_at": datetime.now().strftime("%Y年%m月%d日"),
            "data_source_label": "药品商业数据库公开信息",
            "framework_name": "双轨制敏捷筛选体系"
        },
        "drug": {
            "generic_name_cn": drug_core.get('generic_name_cn', ''),
            "generic_name_en": drug_core.get('generic_name_en', ''),
            "brand_name": drug_core.get('brand_name', ''),
            "brand_name_cn": drug_core.get('brand_name_cn', ''),
            "drug_type": drug_core.get('drug_type', ''),
            "drug_subtype": drug_core.get('drug_subtype', ''),
            "originator": drug_core.get('originator', ''),
            "originator_full": drug_core.get('originator_full', ''),
            "target_name": drug_core.get('target_name', ''),
            "dosage_form": drug_core.get('dosage_form', ''),
            "route_admin": drug_core.get('route_admin', ''),
            "tags_list": tags_list,
            "tagline": tagline,
            "mechanism_desc": drug_core.get('mechanism_desc', ''),
            "mechanism_source": drug_core.get('mechanism_source', ''),
            "highest_phase": drug_core.get('highest_phase', ''),
            "highest_phase_detail": highest_phase_detail,
        },
        "ta_coverage": raw_data.get('ta_coverage', []),
        "transactions": raw_data.get('transactions', []),
        "geo_coverage": raw_data.get('geo_coverage', []),
        "rd_status": raw_data.get('rd_status', []),
        "clinical_data": raw_data.get('clinical_data', []),
        "target_related": raw_data.get('target_related', []),
        "criteria": raw_data.get('criteria', []),
        "analysis": analysis_data
    }
    
    env = jinja2.Environment()
    env.globals['zip'] = zip
    template = env.from_string(TEMPLATE_HTML)
    html = template.render(**template_data)
    
    return html


# ─── Server Entry ───────────────────────────────────────────────────

async def run_server():
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="drug-report-generator",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

def main():
    logger.info("Starting MCP Report Generator Server (stdio)...")
    asyncio.run(run_server())

if __name__ == "__main__":
    main()
