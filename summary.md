# AI 会议系统 — 建设进度总结

> 项目路径：`autogenmeeting1/`  
> 主文件：`main.py`（~1266 行）  
> 规范文档：`construct_manual.md`

---

## 一、系统架构一览

```
MeetingInstance
├── MeetingConfig        # 静态配置（主题 / 议程列表 / 专家列表）
│   ├── AgendaConfig[]   # 每条议程（id / title / goal）
│   └── ExpertConfig[]   # 每位专家（role_name / model / temperature / top_p / seed / system_prompt）
└── MeetingState         # 运行时可变状态
    ├── stage            # 当前阶段（1/2/3）
    ├── agenda_status    # {"A1": "resolved" | "deferred", ...}
    ├── agenda_minutes   # {"A1": AgendaMinutes, ...}
    ├── cross_agenda_minutes  # CrossAgendaMinutes
    ├── deferred_items   # List[DeferredItem]
    ├── user_notes       # 用户补充意见
    └── final_report     # Dict（最终总报告 JSON）
```

---

## 二、已完成模块

### 2.1 数据类（Dataclasses）

| 类名 | 用途 |
|------|------|
| `ExpertConfig` | 专家配置（模型名 / 采样参数 / 系统提示词） |
| `AgendaConfig` | 单条议程（id / 标题 / 目标） |
| `AgendaMinutes` | 单议程纪要结构（conclusion / key_decisions / divergences / risks / assumptions / expert_positions） |
| `CrossAgendaMinutes` | 横向启发纪要（cross_agenda_conflicts / cross_agenda_synergies / topics_for_final_stage） |
| `DeferredItem` | 待定项（id / type / related_agendas / reason / source_stage / resolved） |
| `MeetingConfig` | 会议静态配置 |
| `MeetingState` | 会议运行状态 |
| `MeetingInstance` | 会议实例（config + state） |

---

### 2.2 LLM 调用层

| 函数 | 作用 |
|------|------|
| `get_openai_client()` | 读取 `.env` 构造 `openai.OpenAI` 客户端，Base URL = `https://www.dmxapi.cn/v1` |
| `call_expert_once(expert, agenda)` | 专家立场声明（阶段1第1轮），含异常 fallback |
| `call_expert_debate(expert, agenda, prev_round, round_num)` | 专家辩论（质疑/补充 or 最终表态），基于上一轮所有他人发言 |
| `call_expert_cross_agenda(expert, instance)` | 专家阅读全部单议程纪要后给出跨议程冲突/协同观察（阶段2） |
| `call_secretary_for_minutes(...)` | Secretary LLM 生成单议程 JSON 纪要，缺字段触发一次重写 |
| `call_secretary_for_cross_minutes(...)` | Secretary LLM 汇总专家观察生成跨议程 JSON，缺字段触发一次重写 |
| `call_secretary_for_final_report(instance)` | Secretary LLM 综合全会议内容生成最终总报告 JSON，缺字段触发一次重写 |

所有 LLM 函数均有：
- `try/except` 容错 + 明确 fallback
- ```` ```json ````包裹去除
- `json.loads` 解析 + list 类型校正

---

### 2.3 议程纪要生成（Secretary 强约束）

**单议程纪要（`AgendaMinutes`）必填字段：**
```json
{
  "conclusion": "精炼的一句话总结",
  "key_decisions": ["..."],
  "divergences": ["..."],
  "risks": ["..."],
  "assumptions": ["..."]
}
```

**跨议程纪要（`CrossAgendaMinutes`）必填字段：**
```json
{
  "cross_agenda_conflicts": [
    {"agendas": ["A1","A2"], "description": "...", "severity": "high/medium/low", "suggested_resolution": "..."}
  ],
  "cross_agenda_synergies": [
    {"agendas": ["A1","A2"], "description": "...", "action_items": ["..."]}
  ],
  "topics_for_final_stage": ["..."]
}
```

**最终总报告（`final_report`）必填字段：**
```json
{
  "project_summary": {"processing_date": "...", "total_agendas": N, "resolved_agendas": N, "deferred_with_reason": N},
  "agenda_decisions": [{"agenda_id": "...", "title": "...", "decision": "...", "key_points": [...], "risks": [...]}],
  "cross_agenda_resolutions": [{"issue": "...", "resolution": "...", "affected_agendas": [...]}],
  "overall_risks": [...],
  "key_assumptions": [...],
  "action_items": [{"item": "...", "owner": "...", "deadline": "..."}],
  "deferred_items": [{"id": "...", "type": "...", "related_agendas": [...], "title": "...", "reason": "...", "review_trigger": "..."}]
}
```

---

### 2.4 会议流程函数

| 函数 | 状态 | 说明 |
|------|------|------|
| `run_agenda_session(instance, agenda)` | ✅ 真实 LLM | 单议程子会议：立场声明 → 辩论轮1 → 辩论轮2 → 用户确认 → Secretary 生成纪要 |
| `run_stage_1(instance)` | ✅ | 遍历所有议程，依次调用 `run_agenda_session` |
| `generate_cross_agenda_minutes(instance)` | ✅ 真实 LLM | 专家跨议程观察 + Secretary 汇总，失败时 fallback 规则逻辑 |
| `run_stage_2(instance)` | ✅ 真实 LLM | 汇总阶段1纪要 → 调 `generate_cross_agenda_minutes` → 自动生成跨议程 DeferredItem → 用户确认继续 |
| `run_stage_3(instance)` | ✅ 人工确认 | 按 `cross_agenda → single_agenda` 顺序处理所有待定项，用户逐个输入"共识/继续待定/补充意见" |
| `generate_final_report(instance)` | ✅ 真实 LLM | Secretary 综合所有纪要生成最终总报告，失败时 fallback 拼装 |
| `run_meeting(instance)` | ✅ | 串联全流程：概览 → 阶段1 → 阶段2 → 阶段3 → 总结输出 → 最终总报告 |

---

### 2.5 完整 CLI 输出流（已验证）

```
会议实例概览
  └─ 阶段 1：按议程分组深挖
       ├─ [A1] 立场声明轮 → 辩论轮1 → 辩论轮2 → 用户确认 → Secretary 纪要(LLM)
       └─ [A2] 立场声明轮 → 辩论轮1 → 辩论轮2 → 用户确认 → Secretary 纪要(LLM)
  └─ 阶段 2：全体横向启发
       ├─ 各专家跨议程观察(LLM)
       ├─ Secretary 跨议程纪要(LLM) → JSON
       └─ 自动生成 cross_agenda DeferredItem 写入待定池
  └─ 阶段 3：统一处理待定池
       └─ 逐条人工确认（共识/继续待定/补充意见）
  └─ 会议总结（状态 + 待定项统计 + 阶段1纪要JSON + 阶段2纪要JSON）
  └─ 最终总报告(LLM) → JSON
```

---

## 三、环境配置

**`.env` 关键字段：**
```env
OPENAI_API_KEY=your_dmx_key
OPENAI_API_BASE=https://www.dmxapi.cn/v1   # 必须带 /v1
MODEL_NAME=glm-5
```

> 注意：Base URL 必须是 `https://www.dmxapi.cn/v1`，不带 `/v1` 会返回 HTML 页面导致 `'str' object has no attribute 'choices'`。

**运行方式：**
```powershell
uv run python main.py
```

---

## 四、待建设 / 下一步

| 优先级 | 功能 | 说明 |
|--------|------|------|
| 高 | **最终总报告验证运行** | `generate_final_report` 已实现，尚未在完整流程中观察 LLM 输出 |
| 高 | **持久化（存档/恢复）** | 将 `MeetingInstance` 序列化为 JSON 文件，支持 `--resume` 加载 |
| 中 | **多实例管理** | 按 `construct_manual.md` 中的元级架构，支持新建 / 列出 / 编辑实例 |
| 中 | **专家立场更新机制** | 在关键节点让专家显式声明"立场是否改变"（manual §5.3） |
| 中 | **辩论轮数可配置** | 目前硬编码 2 轮，后续可在 `AgendaConfig` 中配置 `max_debate_rounds` |
| 低 | **AutoGen AgentChat 接入** | `create_moderator_agent` / `create_secretary_agent` 已有骨架，可接入真实多智能体编排 |
| 低 | **阶段3 LLM 辅助** | 目前阶段3全靠人工确认；可让 LLM 为每个待定项给出建议解决方案 |
| 低 | **输出导出** | 会议结束后将所有 JSON 纪要 + 总报告写入文件 |

---

## 五、文件清单

| 文件 | 用途 |
|------|------|
| `main.py` | 全部核心实现（约1266行） |
| `.env` | 本地密钥与接口配置（不入 Git） |
| `.env.example` | 配置模板 |
| `construct_manual.md` | 系统设计规范（三阶段流程 / JSON格式 / 记忆策略） |
| `test_dmx.py` | DMX 接口连通性测试脚本 |
| `test_env.py` | 环境变量加载验证脚本 |
| `pyproject.toml` | uv 依赖管理（autogen-agentchat / openai / python-dotenv） |
| `platforms.md` | 模型平台备注 |
