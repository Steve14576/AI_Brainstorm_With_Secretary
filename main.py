"""Main entry for the multi‑agenda AI meeting system.

This file defines the core skeleton:
- Configuration dataclasses (MeetingConfig/State/Instance)
- Basic expert and agenda configuration
- A stubbed run_meeting function that will later integrate AutoGen + LiteLLM

Current scope (MVP step 1):
- Load environment variables
- Construct a demo MeetingInstance with 2 agendas & a small expert set
- Print a structured overview so we can verify the architecture is wired correctly

Later steps will add:
- AutoGen agents (Moderator/Secretary/Experts)
- Stage 1/2/3 orchestration
- JSON‑structured outputs & deferred pool handling
"""

from __future__ import annotations

import os
import json
import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import OpenAI


# ---------------------------------------------------------------------------
# Dataclass definitions
# ---------------------------------------------------------------------------


@dataclass
class ExpertConfig:
    """Configuration for a single expert agent.

    Notes:
    - `model` is the LiteLLM-exposed model name (which may route to DMXAPI).
    - Generation parameters are per-expert so different styles can be configured.
    """

    role_name: str
    system_prompt: str
    model: str
    temperature: float = 0.7
    top_p: float = 0.9
    seed: Optional[int] = None


@dataclass
class AgendaConfig:
    """Configuration for a single agenda within a meeting."""

    id: str
    title: str
    goal: str
    priority: str = "medium"  # e.g. high / medium / low
    estimated_rounds: int = 3
    expert_ids: List[str] = field(default_factory=list)  # 哪些专家参与本议程（role_name 列表）
    moderator_id: Optional[str] = None  # 可选：该议程主持人 agent id
    secretary_id: Optional[str] = None  # 可选：该议程秘书 agent id


@dataclass
class AgendaMinutes:
    """单个议程的结构化纪要（符合 manual 中的 JSON 输出规范）。"""

    agenda_id: str
    agenda_title: str
    status: str  # "resolved" or "deferred"
    conclusion: str = ""
    key_decisions: List[str] = field(default_factory=list)
    divergences: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    expert_positions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典，方便 JSON 序列化。"""
        return {
            "agenda_id": self.agenda_id,
            "agenda_title": self.agenda_title,
            "status": self.status,
            "conclusion": self.conclusion,
            "key_decisions": self.key_decisions,
            "divergences": self.divergences,
            "risks": self.risks,
            "assumptions": self.assumptions,
            "expert_positions": self.expert_positions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgendaMinutes":
        return cls(
            agenda_id=data.get("agenda_id", ""),
            agenda_title=data.get("agenda_title", ""),
            status=data.get("status", "unknown"),
            conclusion=data.get("conclusion", ""),
            key_decisions=list(data.get("key_decisions", [])),
            divergences=list(data.get("divergences", [])),
            risks=list(data.get("risks", [])),
            assumptions=list(data.get("assumptions", [])),
            expert_positions=dict(data.get("expert_positions", {})),
        )


@dataclass
class CrossAgendaMinutes:
    """阶段 2 横向启发纪要（符合 manual 中的 JSON 输出规范）。"""

    cross_agenda_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    cross_agenda_synergies: List[Dict[str, Any]] = field(default_factory=list)
    topics_for_final_stage: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典，方便 JSON 序列化。"""
        return {
            "cross_agenda_conflicts": self.cross_agenda_conflicts,
            "cross_agenda_synergies": self.cross_agenda_synergies,
            "topics_for_final_stage": self.topics_for_final_stage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossAgendaMinutes":
        return cls(
            cross_agenda_conflicts=list(data.get("cross_agenda_conflicts", [])),
            cross_agenda_synergies=list(data.get("cross_agenda_synergies", [])),
            topics_for_final_stage=list(data.get("topics_for_final_stage", [])),
        )


@dataclass
class DeferredItem:
    """A single deferred item in the unified deferred pool."""

    id: str
    type: str  # "single_agenda" or "cross_agenda"
    related_agendas: List[str]
    reason: str
    source_stage: int
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "related_agendas": self.related_agendas,
            "reason": self.reason,
            "source_stage": self.source_stage,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeferredItem":
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "single_agenda"),
            related_agendas=list(data.get("related_agendas", [])),
            reason=data.get("reason", ""),
            source_stage=int(data.get("source_stage", 1)),
            resolved=bool(data.get("resolved", False)),
        )


@dataclass
class MeetingConfig:
    """Static configuration for a meeting instance.

    This is mostly user-defined metadata and expert/agenda definitions.
    """

    title: str
    agendas: List[AgendaConfig]
    experts: List[ExpertConfig]
    moderator_prompt: str
    secretary_prompt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "agendas": [
                {
                    "id": a.id,
                    "title": a.title,
                    "goal": a.goal,
                    "priority": a.priority,
                    "estimated_rounds": a.estimated_rounds,
                    "expert_ids": a.expert_ids,
                    "moderator_id": a.moderator_id,
                    "secretary_id": a.secretary_id,
                }
                for a in self.agendas
            ],
            "experts": [
                {
                    "role_name": e.role_name,
                    "system_prompt": e.system_prompt,
                    "model": e.model,
                    "temperature": e.temperature,
                    "top_p": e.top_p,
                    "seed": e.seed,
                }
                for e in self.experts
            ],
            "moderator_prompt": self.moderator_prompt,
            "secretary_prompt": self.secretary_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeetingConfig":
        agendas = [
            AgendaConfig(
                id=a.get("id", ""),
                title=a.get("title", ""),
                goal=a.get("goal", ""),
                priority=a.get("priority", "medium"),
                estimated_rounds=int(a.get("estimated_rounds", 3)),
                expert_ids=list(a.get("expert_ids", [])),
                moderator_id=a.get("moderator_id"),
                secretary_id=a.get("secretary_id"),
            )
            for a in data.get("agendas", [])
        ]
        experts = [
            ExpertConfig(
                role_name=e.get("role_name", ""),
                system_prompt=e.get("system_prompt", ""),
                model=e.get("model", ""),
                temperature=float(e.get("temperature", 0.7)),
                top_p=float(e.get("top_p", 0.9)),
                seed=e.get("seed"),
            )
            for e in data.get("experts", [])
        ]
        return cls(
            title=data.get("title", ""),
            agendas=agendas,
            experts=experts,
            moderator_prompt=data.get("moderator_prompt", ""),
            secretary_prompt=data.get("secretary_prompt", ""),
        )


@dataclass
class MeetingState:
    """Mutable state for a meeting instance during execution."""

    stage: int = 1  # 1, 2, or 3
    current_agenda_id: Optional[str] = None
    deferred_items: List[DeferredItem] = field(default_factory=list)
    user_notes: List[str] = field(default_factory=list)
    agenda_status: Dict[str, str] = field(
        default_factory=dict
    )  # e.g. {"A1": "resolved", "A2": "deferred"}
    agenda_minutes: Dict[str, AgendaMinutes] = field(
        default_factory=dict
    )  # e.g. {"A1": AgendaMinutes(...), "A2": ...}
    cross_agenda_minutes: Optional[CrossAgendaMinutes] = None
    final_report: Optional[Dict[str, Any]] = None  # 最终总报告（秘书 LLM 生成）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "current_agenda_id": self.current_agenda_id,
            "deferred_items": [item.to_dict() for item in self.deferred_items],
            "user_notes": list(self.user_notes),
            "agenda_status": dict(self.agenda_status),
            "agenda_minutes": {
                aid: minutes.to_dict() for aid, minutes in self.agenda_minutes.items()
            },
            "cross_agenda_minutes": (
                self.cross_agenda_minutes.to_dict() if self.cross_agenda_minutes else None
            ),
            "final_report": self.final_report,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeetingState":
        state = cls(
            stage=int(data.get("stage", 1)),
            current_agenda_id=data.get("current_agenda_id"),
            deferred_items=[
                DeferredItem.from_dict(d) for d in data.get("deferred_items", [])
            ],
            user_notes=list(data.get("user_notes", [])),
            agenda_status=dict(data.get("agenda_status", {})),
        )
        for aid, minutes_data in data.get("agenda_minutes", {}).items():
            state.agenda_minutes[aid] = AgendaMinutes.from_dict(minutes_data)
        cross_data = data.get("cross_agenda_minutes")
        if cross_data:
            state.cross_agenda_minutes = CrossAgendaMinutes.from_dict(cross_data)
        state.final_report = data.get("final_report")
        return state


@dataclass
class MeetingInstance:
    """A single meeting instance (the "unit" we operate on).

    Later we can persist this object (config + state) for resume/edit.
    """

    id: str
    config: MeetingConfig
    state: MeetingState = field(default_factory=MeetingState)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeetingInstance":
        return cls(
            id=data.get("id", ""),
            config=MeetingConfig.from_dict(data.get("config", {})),
            state=MeetingState.from_dict(data.get("state", {})),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

class MeetingStopException(Exception):
    """Raised when user issues /stop to save and exit the meeting."""


DEFAULT_STATE_PATH = os.getenv("MEETING_SAVE_PATH", os.path.join("output", "last_meeting_state.json"))


def load_meeting_instance(path: str) -> MeetingInstance:
    """Load MeetingInstance from a JSON file created by MeetingInstance.save."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return MeetingInstance.from_dict(data)


# ---------------------------------------------------------------------------
# Global command parsing (Phase 1.3)
# ---------------------------------------------------------------------------


class SkipAgendaException(Exception):
    """Raised when user issues /skip <agenda_id>."""
    def __init__(self, agenda_id: str) -> None:
        self.agenda_id = agenda_id
        super().__init__(agenda_id)


def _handle_global_commands(
    raw: str,
    instance: MeetingInstance,
    current_agenda_id: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Optional[str]:
    """解析全局指令并执行对应行为。

    返回局：
    - None          ：已处理，调用方应根据指令类型決定后续操作（如护照 exception）
    - raw           ：不是全局指令，返回原字符串供调用方处理
    """
    stripped = raw.strip()
    if not stripped.startswith("/"):
        return stripped  # 普通输入，直接返回

    parts = stripped.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/stop":
        print("[/stop] 中止会议并保存进度...")
        sp = save_path or DEFAULT_STATE_PATH
        instance.save(sp)
        print(f"[/stop] 已保存到 {sp}，程序即将退出。")
        raise MeetingStopException()

    if cmd == "/note":
        if arg:
            note = f"[/note] {arg}"
            instance.state.user_notes.append(note)
            print(f"[备注已记录] {arg}")
        else:
            print("[用法] /note <备注内容>")
        # note 后继续当前循环
        return None  # 已处理，不中断主流

    if cmd == "/pause":
        print("[/pause] 已暂停，输入 /resume 继续。")
        while True:
            resume_input = input("> ").strip().lower()
            if resume_input == "/resume":
                print("[/resume] 继续。")
                break
            print("[/pause] 进行中… 输入 /resume 可继续。")
        return None  # 已处理

    if cmd == "/skip":
        target_id = arg or current_agenda_id
        if not target_id:
            print("[用法] /skip <议程编号>，例如 /skip A1")
            return None
        print(f"[/skip] 议程 {target_id} 将被标记为待定，进入阶段 3 处理。")
        raise SkipAgendaException(target_id)

    if cmd == "/redo":
        target_id = arg
        if not target_id:
            print("[用法] /redo <议程编号>，例如 /redo A1")
            return None
        # 清空相关议程数据
        if target_id in instance.state.agenda_minutes:
            del instance.state.agenda_minutes[target_id]
        if target_id in instance.state.agenda_status:
            del instance.state.agenda_status[target_id]
        # 移除对应待定项
        instance.state.deferred_items = [
            item for item in instance.state.deferred_items
            if not (item.type == "single_agenda" and item.related_agendas == [target_id])
        ]
        print(f"[/redo] 议程 {target_id} 数据已清空，请在阶段 1 重新运行该议程（当前版本需手动重启）。")
        return None

    # 未识别的 / 指令，视为普通输入
    return stripped


def prompt_user(
    prompt: str,
    instance: MeetingInstance,
    current_agenda_id: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """包裱 input()，自动处理全局指令并返回处理后的用户输入。

    - 若为全局指令且已处理（如 /note /pause），循环重新要求输入。
    - 若为 /stop 或 /skip ，抛出对应异常。
    - 其他情况返回字符串。
    """
    while True:
        raw = input(prompt).strip()
        result = _handle_global_commands(raw, instance, current_agenda_id, save_path)
        if result is None:
            # 已被处理（/note /pause /redo 等）且不需要中断主流，重新要求输入
            continue
        return result

def export_meeting_outputs(instance: MeetingInstance, output_dir: str = "output") -> None:
    """Export key JSON artifacts for the meeting to the output directory.

    - {instance_id}_stage1_minutes.json
    - {instance_id}_stage2_cross.json
    - {instance_id}_final_report.json
    - {instance_id}_state.json
    """

    os.makedirs(output_dir, exist_ok=True)
    instance_id = instance.id or "meeting"

    # 阶段 1 议程纪要
    minutes_payload: Dict[str, Any] = {
        aid: minutes.to_dict() for aid, minutes in instance.state.agenda_minutes.items()
    }
    with open(
        os.path.join(output_dir, f"{instance_id}_stage1_minutes.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(minutes_payload, f, ensure_ascii=False, indent=2)

    # 阶段 2 横向启发纪要
    if instance.state.cross_agenda_minutes is not None:
        cross_payload = instance.state.cross_agenda_minutes.to_dict()
    else:
        cross_payload = {}
    with open(
        os.path.join(output_dir, f"{instance_id}_stage2_cross.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(cross_payload, f, ensure_ascii=False, indent=2)

    # 最终总报告
    final_report = instance.state.final_report or {}
    with open(
        os.path.join(output_dir, f"{instance_id}_final_report.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    # 完整实例快照（用于恢复）
    state_path = os.path.join(output_dir, f"{instance_id}_state.json")
    instance.save(state_path)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def load_environment() -> None:
    """Load environment variables from .env and validate basics.

    For now we only check that the main model name is present. LiteLLM/DMXAPI
    specific settings (base URL, API keys) are expected to be configured via
    environment variables but are not validated here.
    """

    load_dotenv()

    model_main = os.getenv("MEETING_MAIN_MODEL")
    if not model_main:
        # 兼容直接使用 DMX 的 MODEL_NAME 配置
        model_main = os.getenv("MODEL_NAME")

    if not model_main:
        print("[警告] 未配置 MEETING_MAIN_MODEL / MODEL_NAME，后续将无法真正调用模型。")
        print("       请在 .env 中设置，例如: MEETING_MAIN_MODEL=glm-5 或 MODEL_NAME=glm-5")
    else:
        print(f"[环境] 主模型 = {model_main}")


def build_llm_config_for_expert(expert: ExpertConfig) -> Dict[str, Any]:
    """构造采样相关配置（目前只用于记录，实际采样参数通过 ModelClient 传递）。"""

    return {
        "temperature": expert.temperature,
        "top_p": expert.top_p,
        "seed": expert.seed,
    }


def get_openai_client() -> OpenAI:
    """构造一个直接调用 OpenAI 兼容接口的客户端（用于 DMX/LiteLLM）。"""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    if not api_key:
        print("[警告] 未配置 OPENAI_API_KEY，模型调用很可能会失败。")
    if not base_url:
        print("[警告] 未配置 OPENAI_API_BASE，将使用默认 OpenAI 地址，这通常不符合 DMX 场景。")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_expert_once(expert: ExpertConfig, agenda: AgendaConfig) -> str:
    """调用一次专家模型，返回立场声明文本。

    为了先跑通主流程，这里直接用 openai 客户端调用 /v1/chat/completions，
    不再通过 ModelClient 的高级抽象。
    如果调用失败，返回错误信息。
    """

    try:
        client = get_openai_client()
        system_prompt = expert.system_prompt
        user_message = (
            "你正在参加一个多议程策略会议。当前议程是：" f"{agenda.title}。"\
            "请以该角色的视角，简要给出一段立场声明（不超过 150 字），"\
            "说明你认为本议程最关键要解决的问题是什么。"
        )

        resp = client.chat.completions.create(
            model=expert.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=expert.temperature,
            top_p=expert.top_p,
        )
        content = resp.choices[0].message.content or ""
        # OpenAI 新 SDK 返回的 content 可能是 str 或 list，根据 DMX 实际情况做简单兼容
        if isinstance(content, list):
            # 拼接所有文本片段
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content.strip()
    except Exception as e:
        error_msg = f"[模型调用失败] {expert.role_name} - {type(e).__name__}: {str(e)[:100]}"
        print(f"\n{error_msg}")
        return f"(模型调用失败，占位立场声明)"


def call_expert_debate(expert: ExpertConfig, agenda: AgendaConfig,
                       prev_round: Dict[str, str], round_num: int) -> str:
    """调用专家模型，进行一轮辩论回应。

    prev_round: 其他专家在上一轮的发言（role_name -> 内容）。
    round_num:  辩论轮次编号（第 1/2 轮），用于提示词中提示。
    """
    try:
        client = get_openai_client()
        
        # 构建其他专家已表态的数句摘要
        others_text = "\n\n".join(
            f"[{name}的上轮发言]\n{text}"
            for name, text in prev_round.items()
            if name != expert.role_name
        )
        if not others_text:
            others_text = "(暂无其他专家发言)"
        
        round_tip = "请針对其中的关键观点进行质疑或补充" if round_num == 1 \
                    else "箱如各专家已进行多轮辩论，请给出你最终的立场有无改变及理由"
        
        user_message = (
            f"当前议程：{agenda.title}\n\n"
            f"其他专家的发言：\n{others_text}\n\n"
            f"现在是第 {round_num} 轮辩论。{round_tip}，"
            f"不超过 150 字，直接输出回应内容。"
        )
        
        resp = client.chat.completions.create(
            model=expert.model,
            messages=[
                {"role": "system", "content": expert.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=expert.temperature,
            top_p=expert.top_p,
        )
        content = resp.choices[0].message.content or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content.strip()
    except Exception as e:
        print(f"\n[辩论轮{round_num}调用失败] {expert.role_name} - {type(e).__name__}: {str(e)[:100]}")
        return f"(辩论回应调用失败，占位)"


def create_moderator_agent(config: MeetingConfig) -> AssistantAgent:
    """Create the moderator agent from MeetingConfig (AgentChat API)."""

    model_main = os.getenv("MEETING_MAIN_MODEL") or os.getenv("MODEL_NAME", "glm-5")
    sampling: Dict[str, Any] = {"temperature": 0.4, "top_p": 0.9}
    model_client = build_model_client(model_main, sampling)
    return AssistantAgent(
        name="moderator",
        system_message=config.moderator_prompt,
        model_client=model_client,
    )


def create_secretary_agent(config: MeetingConfig) -> AssistantAgent:
    """Create the secretary agent (AgentChat API).

    按 manual 要求，秘书必须输出 JSON，对后续的纪要/待定池处理非常关键。
    """

    model_main = os.getenv("MEETING_MAIN_MODEL") or os.getenv("MODEL_NAME", "glm-5")
    sampling: Dict[str, Any] = {"temperature": 0.2, "top_p": 0.9}
    model_client = build_model_client(model_main, sampling)
    # JSON / 结构化输出在后续 prompt 中通过 response_format 或系统指令约束
    return AssistantAgent(
        name="secretary",
        system_message=config.secretary_prompt,
        model_client=model_client,
    )


# ---------------------------------------------------------------------------
# Configuration builders
# ---------------------------------------------------------------------------


def build_demo_meeting_config() -> MeetingConfig:
    """Build a small demo MeetingConfig used for the first MVP.

    - 2 agendas: 产品方向 / 技术方向
    - 2 experts: 一个产品专家 + 一个技术专家
    - 1 moderator + 1 secretary (prompts only, agent creation to be added later)
    """

    model_main = os.getenv("MEETING_MAIN_MODEL") or os.getenv("MODEL_NAME", "glm-5")

    agendas = [
        AgendaConfig(
            id="A1",
            title="产品方向与目标用户",
            goal="澄清本产品的目标用户画像与核心价值主张",
            priority="high",
            estimated_rounds=3,
            expert_ids=["product_expert_1", "tech_expert_1"],
            moderator_id="moderator",
            secretary_id="secretary",
        ),
        AgendaConfig(
            id="A2",
            title="技术路线与实现边界",
            goal="确定 MVP 阶段的技术栈与实现范围",
            priority="high",
            estimated_rounds=3,
            expert_ids=["product_expert_1", "tech_expert_1"],
            moderator_id="moderator",
            secretary_id="secretary",
        ),
    ]

    experts = [
        ExpertConfig(
            role_name="product_expert_1",
            system_prompt=(
                "你是一名产品负责人，擅长从用户和业务角度思考。"
                "在会议中，你重点关注用户价值、需求优先级和可行的产品方案。"
            ),
            model=model_main,
            temperature=0.7,
            top_p=0.9,
        ),
        ExpertConfig(
            role_name="tech_expert_1",
            system_prompt=(
                "你是一名务实的技术负责人，擅长在有限资源下做出合理技术决策。"
                "在会议中，你重点关注技术可行性、风险与实现路径。"
            ),
            model=model_main,
            temperature=0.6,
            top_p=0.9,
        ),
    ]

    moderator_prompt = (
        "你是一位严谨的会议主持人，负责拆解议程、控制节奏并推动收敛。"
        "你会根据当前议程目标，提醒专家聚焦关键问题，避免跑题。"
    )

    secretary_prompt = (
        "你是一位会议秘书，只负责记录与总结，不参与决策。"
        "你必须用结构化 JSON 输出纪要，标注结论、分歧、风险和待定项。"
    )

    return MeetingConfig(
        title="Demo 多议程 AI 会议",
        agendas=agendas,
        experts=experts,
        moderator_prompt=moderator_prompt,
        secretary_prompt=secretary_prompt,
    )


def build_meeting_config_from_files(agents_path: str, agendas_path: str) -> MeetingConfig:
    """从 JSON 文件构建 MeetingConfig。

    agents.json: 定义所有 agent 的模型参数与 system prompt；
    agendas.json: 定义议程列表及每个议程的参与者（哪些 expert、主持人、秘书）。
    """

    # 读取 agents.json
    with open(agents_path, "r", encoding="utf-8") as f:
        agents_raw = json.load(f)
    agents_list = agents_raw.get("agents", [])
    agent_map: Dict[str, Dict[str, Any]] = {a.get("id"): a for a in agents_list if a.get("id")}

    # 读取 agendas.json
    with open(agendas_path, "r", encoding="utf-8") as f:
        agendas_raw = json.load(f)

    title = agendas_raw.get("title", "多议程 AI 会议")
    agendas_cfg: List[AgendaConfig] = []
    for a in agendas_raw.get("agendas", []):
        participants = a.get("participants", {}) or {}
        expert_ids = list(participants.get("experts", []))
        moderator_id = participants.get("moderator")
        secretary_id = participants.get("secretary")
        agendas_cfg.append(
            AgendaConfig(
                id=a.get("id", ""),
                title=a.get("title", ""),
                goal=a.get("goal", ""),
                priority=a.get("priority", "medium"),
                estimated_rounds=int(a.get("estimated_rounds", 3)),
                expert_ids=expert_ids,
                moderator_id=moderator_id,
                secretary_id=secretary_id,
            )
        )

    # 根据 agents.json 中 kind == "expert" 的条目构建 ExpertConfig
    experts_cfg: List[ExpertConfig] = []
    for agent_id, agent in agent_map.items():
        if agent.get("kind", "expert") != "expert":
            continue
        experts_cfg.append(
            ExpertConfig(
                role_name=agent_id,
                system_prompt=agent.get("system_prompt", ""),
                model=agent.get("model", os.getenv("MEETING_MAIN_MODEL") or os.getenv("MODEL_NAME", "glm-5")),
                temperature=float(agent.get("temperature", 0.7)),
                top_p=float(agent.get("top_p", 0.9)),
                seed=agent.get("seed"),
            )
        )

    # 从 agents.json 中抽取全局主持人/秘书的 system prompt（若存在）
    moderator_prompt = "你是一位严谨的会议主持人，负责拆解议程、控制节奏并推动收敛。"
    secretary_prompt = "你是一位会议秘书，只负责记录与总结，不参与决策。你必须用结构化 JSON 输出纪要，标注结论、分歧、风险和待定项。"
    moderator_def = agent_map.get("moderator")
    if moderator_def and moderator_def.get("system_prompt"):
        moderator_prompt = moderator_def["system_prompt"]
    secretary_def = agent_map.get("secretary")
    if secretary_def and secretary_def.get("system_prompt"):
        secretary_prompt = secretary_def["system_prompt"]

    return MeetingConfig(
        title=title,
        agendas=agendas_cfg,
        experts=experts_cfg,
        moderator_prompt=moderator_prompt,
        secretary_prompt=secretary_prompt,
    )


# ---------------------------------------------------------------------------
# Meeting runner (skeleton)
# ---------------------------------------------------------------------------


def create_demo_instance(instance_id: str = "demo-001") -> MeetingInstance:
    """Create a MeetingInstance.

    优先从 JSON 配置文件（agents.json / agendas.json）构建；
    若文件不存在，则退回到内置的 demo 配置。
    """

    agents_path = "agents.json"
    agendas_path = "agendas.json"
    if os.path.exists(agents_path) and os.path.exists(agendas_path):
        config = build_meeting_config_from_files(agents_path, agendas_path)
    else:
        config = build_demo_meeting_config()
    state = MeetingState(stage=1, current_agenda_id=config.agendas[0].id if config.agendas else None)
    return MeetingInstance(id=instance_id, config=config, state=state)


# ---------------------------------------------------------------------------
# Meeting runner (stage orchestration)
# ---------------------------------------------------------------------------


def call_secretary_for_minutes(
    agenda: AgendaConfig,
    position_statements: Dict[str, str],
    debate_rounds: List[Dict[str, str]],
    user_comments: List[str],
    status: str,
) -> Optional[Dict[str, Any]]:
    """调用秘书 LLM ，根据全程讨论生成结构化议程纪要 JSON。

    输入：全程所有轮次的专家发言 + 用户意见 + 最终状态。
    输出：字典，字段包含 conclusion / key_decisions / divergences / risks / assumptions。
    失败时返回 None（由调用方 fallback）。
    """
    REQUIRED_FIELDS = {"conclusion", "key_decisions", "divergences", "risks", "assumptions"}

    def _build_transcript() -> str:
        lines = []
        lines.append(f"议程：{agenda.title}")
        lines.append(f"议程目标：{agenda.goal}")
        lines.append(f"最终状态：{status}")
        if user_comments:
            lines.append(f"用户补充意见：{'; '.join(user_comments)}")
        lines.append("\n[立场声明轮]")
        for name, text in position_statements.items():
            lines.append(f"  {name}: {text}")
        for idx, rnd in enumerate(debate_rounds, start=1):
            round_label = "最终表态" if idx == len(debate_rounds) else "质疑与补充"
            lines.append(f"\n[辩论轮 {idx}：{round_label}]")
            for name, text in rnd.items():
                lines.append(f"  {name}: {text}")
        return "\n".join(lines)

    system_prompt = (
        "你是一名会议秘书，擅长提炼讨论内容并输出精准的结构化纪要。\n"
        "你的回应必须是且仅是一个合法的 JSON 对象，"
        "不包含任何额外解释、自然语言、代码块标记。\n"
        "输出格式：\n"
        '{\n'
        '  "conclusion": "总结性结论（一句话）",\n'
        '  "key_decisions": ["关键决策 1", "关键决策 2"],\n'
        '  "divergences": ["分歧点 1"],\n'
        '  "risks": ["风险 1"],\n'
        '  "assumptions": ["假设前提 1"]\n'
        '}'
    )

    transcript = _build_transcript()
    user_message = (
        f"以下是本议程的全程记录：\n\n{transcript}\n\n"
        "请根据上述记录生成纪要 JSON。"
        "注意: 如果议程状态为 resolved，请在 key_decisions 中列出具体共识点；"
        "如果为 deferred，请在 divergences 中列出尚未解决的分歧点。"
    )

    def _call_and_parse(user_msg: str) -> Optional[Dict[str, Any]]:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "glm-5"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            top_p=0.9,
        )
        raw = resp.choices[0].message.content or ""
        # 去掉可能的 ```json ... ``` 包裹
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None

    try:
        result = _call_and_parse(user_message)
        if result is None:
            return None

        # 字段校验：缺少必填字段就重写一次
        missing = REQUIRED_FIELDS - set(result.keys())
        if missing:
            print(f"  [秘书] 缺少字段 {missing}，触发重写...")
            retry_msg = (
                user_message + f"\n\n上次输出缺少必填字段：{missing}，请重新输出完整的 JSON。"
            )
            result = _call_and_parse(retry_msg)
            if result is None or (REQUIRED_FIELDS - set(result.keys())):
                return None

        # 类型修正：确保 list 字段为 list
        for field in ("key_decisions", "divergences", "risks", "assumptions"):
            if not isinstance(result.get(field), list):
                result[field] = [str(result[field])] if result.get(field) else []

        return result

    except Exception as e:
        print(f"  [秘书] JSON 解析失败 ({type(e).__name__}: {str(e)[:60]})，使用简化逻辑生成纪要。")
        return None


def call_secretary_draft_summary(
    agenda: AgendaConfig,
    position_statements: Dict[str, str],
    debate_rounds: List[Dict[str, str]],
    user_comments: List[str],
) -> tuple:
    """秘书出一份简明的议程草稿摘要，包含建议状态和主要分歧点。

    返回 (summary_text: str, recommended_status: str)。
    summary_text 是展示给用户的自然语言摘要（非 JSON，不超过 200 字）。
    recommended_status 为 'resolved' 或 'deferred'。
    """

    def _build_transcript() -> str:
        lines = []
        lines.append(f"议程：{agenda.title}")
        lines.append(f"目标：{agenda.goal}")
        if user_comments:
            lines.append(f"用户补充意见：{'; '.join(user_comments)}")
        lines.append("\n[立场声明轮]")
        for name, text in position_statements.items():
            lines.append(f"  {name}: {text}")
        for idx, rnd in enumerate(debate_rounds, start=1):
            round_label = "最终表态" if idx == len(debate_rounds) else "质疑与补充"
            lines.append(f"\n[辩论轮 {idx}：{round_label}]")
            for name, text in rnd.items():
                lines.append(f"  {name}: {text}")
        return "\n".join(lines)

    system_prompt = (
        "你是一名会议秘书。请对以下议程讨论进行简明草稿，要求："
        "1) 不超过 200 字的自然语言总结（点出核心共识或主要分歧）；"
        "2) 建议一个最终状态：\"resolved\"(已达共识) 或 \"deferred\"(尚存分歧建议待定)；"
        "3) 输出格式为纯文本，不需要 JSON；"
        "4) 最后一行单独写：\"[RECOMMENDED: resolved]\" 或 \"[RECOMMENDED: deferred]\"。"
    )
    transcript = _build_transcript()
    user_message = f"以下是本议程的全程记录：\n\n{transcript}\n\n请将联绪纪要草稿。"

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "glm-5"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            top_p=0.9,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # 提取建议状态
        recommended = "deferred"
        if "[RECOMMENDED: resolved]" in raw:
            recommended = "resolved"
        elif "[RECOMMENDED: deferred]" in raw:
            recommended = "deferred"
        # 移除标记行，保留总结文本
        summary = raw.replace("[RECOMMENDED: resolved]", "").replace("[RECOMMENDED: deferred]", "").strip()
        return summary, recommended
    except Exception as e:
        # Fallbackï¼简单占位
        fallback = f"秘书草稿获取失败 ({type(e).__name__}: {str(e)[:60]})"
        return fallback, "deferred"


def generate_agenda_minutes(
    instance: MeetingInstance,
    agenda: AgendaConfig,
    position_statements: Dict[str, str],
    debate_rounds: List[Dict[str, str]],
    user_comments: List[str],
) -> AgendaMinutes:
    """生成单个议程的结构化纪要。

    优先调用秘书 LLM 生成 JSON，失败时 fallback 到简化逻辑。
    expert_positions 封存最新轮次的立场。
    """

    state = instance.state
    status = state.agenda_status.get(agenda.id, "unknown")

    # expert_positions 封存最终话语（最后一轮辩论优先）
    final_positions = {**position_statements}
    if debate_rounds:
        final_positions.update(debate_rounds[-1])

    expert_positions_summary: Dict[str, str] = {
        name: (text[:120] + "..." if len(text) > 120 else text)
        for name, text in final_positions.items()
    }

    print(f"  [秘书] 正在生成 {agenda.id} 议程纪要...", end="", flush=True)
    llm_result = call_secretary_for_minutes(
        agenda, position_statements, debate_rounds, user_comments, status
    )

    if llm_result:
        print(" 完成（LLM）。")
        return AgendaMinutes(
            agenda_id=agenda.id,
            agenda_title=agenda.title,
            status=status,
            conclusion=str(llm_result.get("conclusion", "")),
            key_decisions=llm_result.get("key_decisions", []),
            divergences=llm_result.get("divergences", []),
            risks=llm_result.get("risks", []),
            assumptions=llm_result.get("assumptions", []),
            expert_positions=expert_positions_summary,
        )

    # Fallback
    print(" 使用简化逻辑。")
    conclusion = (
        f"议程 {agenda.id} 已达成共识，目标为：{agenda.goal}。" if status == "resolved"
        else f"议程 {agenda.id} 暂缓，尚未完全达成共识。"
    )
    divergences = []
    if status == "deferred":
        divergences = ["待定议程，需要进一步讨论。"]
    return AgendaMinutes(
        agenda_id=agenda.id,
        agenda_title=agenda.title,
        status=status,
        conclusion=conclusion,
        key_decisions=([f"用户补充意见：{'; '.join(user_comments)}"] if user_comments else []),
        divergences=divergences,
        risks=[],
        assumptions=[],
        expert_positions=expert_positions_summary,
    )



def run_agenda_session(instance: MeetingInstance, agenda: AgendaConfig) -> None:
    """运行单个议程的阶段 1 子会议流程。

    流程：
    1. 立场声明轮：每个专家独立给出初始表态；
    2. N 轮辩论（由 agenda.estimated_rounds - 1 决定）；
    3. 秘书草稿摘要 + 建议状态，展示给用户；
    4. 用户确认循环：
       - '共识' → 标记 resolved；
       - '待定' → 标记 deferred；
       - 其他文本 → 专家针对用户意见重新辩论一轮，秘书再次草稿，循环直到用户选定共识或待定。
    """

    state = instance.state
    state.current_agenda_id = agenda.id

    print(f"\n===== 开始议程 {agenda.id}: {agenda.title} =====")
    print(f"议程目标: {agenda.goal}")

    # 根据 agenda.expert_ids 选择参与本议程的专家；若为空则退回到全体专家
    if agenda.expert_ids:
        experts_for_agenda = [
            exp for exp in instance.config.experts if exp.role_name in agenda.expert_ids
        ]
    else:
        experts_for_agenda = list(instance.config.experts)

    # ---------- 立场声明轮 ----------
    print("\n-- 立场声明轮 --")
    position_statements: Dict[str, str] = {}
    for expert in experts_for_agenda:
        print(f"\n[专家 {expert.role_name}] 立场声明：")
        statement = call_expert_once(expert, agenda)
        print(statement)
        position_statements[expert.role_name] = statement

    # ---------- 初始辩论轮（Phase 1.4）----------
    num_debate_rounds = max(1, agenda.estimated_rounds - 1)
    debate_rounds: List[Dict[str, str]] = []

    def _run_one_debate_round(
        prev_speeches: Dict[str, str],
        round_idx: int,
        total_rounds: int,
        user_feedback: Optional[str] = None,
    ) -> Dict[str, str]:
        """run one debate round for experts_for_agenda; inject optional user_feedback."""
        is_last = (round_idx == total_rounds)
        label = "最终表态" if is_last else "质疑与补充"
        print(f"\n-- 辩论轮 {round_idx}：{label} --")
        if user_feedback:
            print(f"  [用户进一步意见] {user_feedback}")
        cur: Dict[str, str] = {}
        for expert in experts_for_agenda:
            print(f"\n[专家 {expert.role_name}] 辩论回应：")
            # 如果有用户意见，将其作为一条额外上下文注入给专家
            context = dict(prev_speeches)
            if user_feedback:
                context["[用户进一步意见]"] = user_feedback
            response = call_expert_debate(expert, agenda, context, round_num=round_idx)
            print(response)
            cur[expert.role_name] = response
        return cur

    if len(position_statements) > 1:
        for round_idx in range(1, num_debate_rounds + 1):
            is_last_round = (round_idx == num_debate_rounds)
            prev = debate_rounds[-1] if debate_rounds else position_statements
            cur = _run_one_debate_round(prev, round_idx, num_debate_rounds)
            debate_rounds.append(cur)
            if not is_last_round:
                mod_summary = call_moderator_round_summary(
                    agenda, cur, round_num=round_idx, total_rounds=num_debate_rounds
                )
                if mod_summary:
                    print(f"\n[主持人小结 轮{round_idx}]\n{mod_summary}")

    # ---------- 秘书草稿→用户确认循环（闭环核心）----------
    comments: List[str] = []
    feedback_round_count = 0   # 用户反馈引发的额外辩论轮数

    while True:
        # 秘书生成本轮草稿
        print(f"\n[秘书] 正在绘制本议程摘要…", end="", flush=True)
        draft_text, recommended = call_secretary_draft_summary(
            agenda, position_statements, debate_rounds, comments
        )
        recommended_cn = "共识已可" if recommended == "resolved" else "建议待定"
        print(f" 完成。")
        print(f"\n[秘书草稿] 「{recommended_cn}」\n{draft_text}")

        # 询问用户
        user_input = prompt_user(
            f"\n[{agenda.id}] 请确认本议程状态：共识 / 待定 / 输入其他内容可让专家继续讨论：",
            instance,
            current_agenda_id=agenda.id,
        )

        if user_input == "共识":
            print(f"[{agenda.id}] 已标记为：共识（resolved）")
            state.agenda_status[agenda.id] = "resolved"
            break

        if user_input == "待定":
            print(f"[{agenda.id}] 已标记为：待定（deferred）")
            state.agenda_status[agenda.id] = "deferred"
            reason_text = (
                "\n".join(comments)
                if comments
                else "用户在阶段 1 将本议程标记为待定。"
            )
            deferred_id = f"{agenda.id}-d-{len(state.deferred_items) + 1}"
            state.deferred_items.append(
                DeferredItem(
                    id=deferred_id,
                    type="single_agenda",
                    related_agendas=[agenda.id],
                    reason=reason_text,
                    source_stage=1,
                    resolved=False,
                )
            )
            break

        # 其他输入：视为用户意见，触发专家继续辩论
        feedback_round_count += 1
        print(f"[{agenda.id}] 收到你的进一步意见，专家将进行继续辩论……")
        comments.append(user_input)
        state.user_notes.append(f"[{agenda.id}] 用户进一步意见: {user_input}")

        # 专家针对用户意见进行一轮新辩论
        prev_ctx = debate_rounds[-1] if debate_rounds else position_statements
        new_round = _run_one_debate_round(
            prev_ctx,
            round_idx=num_debate_rounds + feedback_round_count,
            total_rounds=num_debate_rounds + feedback_round_count,
            user_feedback=user_input,
        )
        debate_rounds.append(new_round)
        # 循环回到开头，秘书重新草稿

    # ---------- 生成议程结构化纪要 ----------
    minutes = generate_agenda_minutes(
        instance, agenda, position_statements, debate_rounds, comments
    )
    state.agenda_minutes[agenda.id] = minutes
    print(f"[{agenda.id}] 议程纪要已完成。")

    # Phase 2.1: 专家立场更新
    run_expert_position_updates(instance, agenda)


def run_stage_1(instance: MeetingInstance) -> None:
    """阶段 1：按议程分组深挖（简化版，仅跑子会议骨架）。"""

    print("\n>>>> 阶段 1：按议程分组深入讨论 <<<<")
    for agenda in instance.config.agendas:
        run_agenda_session(instance, agenda)


def _build_minutes_context(instance: MeetingInstance) -> str:
    """将阶段 1 所有议程纪要构建为文本上下文，供专家和秘书阅读。"""
    lines = [f"会议主题：{instance.config.title}"]
    for agenda in instance.config.agendas:
        minutes = instance.state.agenda_minutes.get(agenda.id)
        if not minutes:
            continue
        m = minutes.to_dict()
        lines.append(f"\n=== 议程 {agenda.id}: {agenda.title} (状态: {m['status']}) ===")
        lines.append(f"结论: {m['conclusion']}")
        if m["key_decisions"]:
            lines.append("关键决策: " + "; ".join(m["key_decisions"]))
        if m["divergences"]:
            lines.append("分歧点: " + "; ".join(m["divergences"]))
        if m["risks"]:
            lines.append("风险: " + "; ".join(m["risks"]))
    return "\n".join(lines)


def call_expert_cross_agenda(expert: ExpertConfig, instance: MeetingInstance) -> str:
    """请专家阅读全部阶段 1 议程纪要，给出跨议程观察。"""
    try:
        client = get_openai_client()
        ctx = _build_minutes_context(instance)
        user_message = (
            f"以下是本次会议阶段 1 所有议程的纪要：\n\n{ctx}\n\n"
            "作为该领域专家，请指出不同议程之间可能存在的：\n"
            "1. 冲突：一个议程的决策与另一个议程的决策之间的矛盾\n"
            "2. 协同：不同议程的共识点可以如何协同推进\n"
            "请简要表述，不超过 200 字，直接输出观察内容，无需标题。"
        )
        resp = client.chat.completions.create(
            model=expert.model,
            messages=[
                {"role": "system", "content": expert.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=expert.temperature,
            top_p=expert.top_p,
        )
        content = resp.choices[0].message.content or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content.strip()
    except Exception as e:
        print(f"\n[跨议程观察调用失败] {expert.role_name} - {type(e).__name__}: {str(e)[:80]}")
        return "(占位跨议程观察)"


def call_secretary_for_cross_minutes(
    expert_observations: Dict[str, str],
    instance: MeetingInstance,
) -> Optional[Dict[str, Any]]:
    """秘书 LLM 汇总所有专家跨议程观察，生成结构化 CrossAgendaMinutes JSON。"""
    REQUIRED = {"cross_agenda_conflicts", "cross_agenda_synergies", "topics_for_final_stage"}

    ctx = _build_minutes_context(instance)
    obs_text = "\n\n".join(
        f"[{name} 的跨议程观察]\n{obs}"
        for name, obs in expert_observations.items()
    )
    agenda_ids = [a.id for a in instance.config.agendas]

    system_prompt = (
        "你是会议秘书，擅长整合多方观察并输出精准的 JSON 纪要。\n"
        "你的回应必须是且仅是一个合法的 JSON 对象，不包含任何额外文字、代码块标记。\n"
        "输出格式：\n"
        '{\n'
        '  "cross_agenda_conflicts": [\n'
        '    {"agendas": ["议程ID1", "议程ID2"], "description": "冲突描述",'
        ' "severity": "high/medium/low", "suggested_resolution": "建议解决方式"}\n'
        '  ],\n'
        '  "cross_agenda_synergies": [\n'
        '    {"agendas": ["议程ID1", "议程ID2"], "description": "协同描述",'
        ' "action_items": ["行动项"]}\n'
        '  ],\n'
        '  "topics_for_final_stage": ["阶段 3 需处理的问题"]\n'
        '}'
    )
    user_message = (
        f"阶段 1 议程纪要：\n{ctx}\n\n"
        f"各专家跨议程观察：\n{obs_text}\n\n"
        f"议程列表：{agenda_ids}\n\n"
        "请生成跨议程分析 JSON。如没有真实冲突/协同，对应字段留空列表即可。"
    )

    def _call_and_parse(msg: str) -> Optional[Dict[str, Any]]:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "glm-5"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ],
            temperature=0.2,
            top_p=0.9,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None

    try:
        result = _call_and_parse(user_message)
        if result is None:
            return None
        missing = REQUIRED - set(result.keys())
        if missing:
            print(f"  [秘书] 跨议程纪要缺少字段 {missing}，触发重写...")
            result = _call_and_parse(
                user_message + f"\n\n上次缺少必填字段：{missing}，请重新输出完整 JSON。"
            )
            if result is None or (REQUIRED - set(result.keys())):
                return None
        for field in ("cross_agenda_conflicts", "cross_agenda_synergies", "topics_for_final_stage"):
            if not isinstance(result.get(field), list):
                result[field] = []
        return result
    except Exception as e:
        print(f"  [秘书] 跨议程 JSON 解析失败 ({type(e).__name__}: {str(e)[:60]})，使用简化逻辑。")
        return None


# ---------------------------------------------------------------------------
# Phase 2: AI quality improvements
# ---------------------------------------------------------------------------


def call_expert_position_update(
    expert: ExpertConfig,
    agenda: AgendaConfig,
    minutes: "AgendaMinutes",
) -> Optional[Dict[str, Any]]:
    """议程结束后，要求专家输出结构化立场更新 JSON。

    返回字段： expert_id / support_current_plan / reservations /
                suggest_defer / reasoning
    失败时返回 None。
    """
    REQUIRED = {"expert_id", "support_current_plan", "reservations", "suggest_defer", "reasoning"}
    try:
        client = get_openai_client()
        conclusion_text = f"结论：{minutes.conclusion}" if minutes.conclusion else "(暂无结论)"
        key_dec = "\n".join(f"  - {k}" for k in minutes.key_decisions[:3]) or "  (无)"
        system_prompt = expert.system_prompt
        user_message = (
            f"本议程 '{agenda.title}' 已结束，当前纪要摘要：\n{conclusion_text}\n主要决策：\n{key_dec}\n\n"
            "作为该议程的参与专家，请用 JSON 格式输出你对该议程最终决议的立场：\n"
            '{"expert_id": "<你的角色名>", "support_current_plan": true/false, '
            '"reservations": ["<保留意见>"], "suggest_defer": false, "reasoning": "<理由>"}\n'
            "只输出 JSON，不附其他文字。"
        )
        resp = client.chat.completions.create(
            model=expert.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            top_p=0.9,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not (REQUIRED <= set(parsed.keys())):
            return None
        return parsed
    except Exception as e:
        print(f"  [立场更新失败] {expert.role_name} - {type(e).__name__}: {str(e)[:60]}")
        return None


def run_expert_position_updates(
    instance: MeetingInstance,
    agenda: AgendaConfig,
) -> None:
    """议程结束后调用各专家输出结构化立场更新，结果写入 AgendaMinutes.expert_positions。"""
    minutes = instance.state.agenda_minutes.get(agenda.id)
    if not minutes:
        return
    print(f"\n  [阶段 2.1] 请各专家对议程 {agenda.id} 做最终立场表态...")
    for expert in instance.config.experts:
        result = call_expert_position_update(expert, agenda, minutes)
        if result:
            support = result.get("support_current_plan", True)
            reservations = result.get("reservations", [])
            reasoning = result.get("reasoning", "")
            summary = f"[{'support' if support else 'oppose'}] {reasoning[:100]}"
            if reservations:
                summary += f" | 保留意见: {', '.join(reservations[:2])}"
            minutes.expert_positions[expert.role_name] = summary
            print(f"    [{expert.role_name}] 立场已更新。")
        else:
            print(f"    [{expert.role_name}] 立场更新失败，保留辩论轮原文。")


def call_expert_deferred_suggestion(
    item: DeferredItem,
    instance: MeetingInstance,
) -> str:
    """阶段 3 LLM 辅助：让相关专家基于全局视角给出建议解决方案。

    返回文本摘要，失败时返回空字符串。
    """
    ctx = _build_minutes_context(instance)
    related_ids = item.related_agendas
    relevant_experts = [
        e for e in instance.config.experts
    ]  # 简化：所有专家都展示建议，可后续按相关领域过滤
    suggestions: List[str] = []
    for expert in relevant_experts:
        try:
            client = get_openai_client()
            user_message = (
                f"以下是本次会议所有议程的纪要：\n\n{ctx}\n\n"
                f"当前待定项类型：{item.type}，相关议程：{', '.join(related_ids)}，"
                f"待定原因：{item.reason[:100]}\n\n"
                "基于全局视角，请给出一个具体的建议解决方案（不超过 120 字，直接输出建议内容）。"
            )
            resp = client.chat.completions.create(
                model=expert.model,
                messages=[
                    {"role": "system", "content": expert.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.5,
                top_p=0.9,
            )
            text = (resp.choices[0].message.content or "").strip()
            if isinstance(text, list):
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            if text:
                suggestions.append(f"[{expert.role_name}] {text}")
        except Exception as e:
            print(f"  [建议调用失败] {expert.role_name}: {str(e)[:60]}")
    return "\n".join(suggestions)


def call_moderator_round_summary(
    agenda: AgendaConfig,
    round_speeches: Dict[str, str],
    round_num: int,
    total_rounds: int,
) -> str:
    """主持人辩论轮小结：主持人读取本轮所有专家发言，输出分歧摘要和下轮聚焦建议。

    返回结果文本，失败时返回空字符串。
    """
    model_main = os.getenv("MEETING_MAIN_MODEL") or os.getenv("MODEL_NAME", "glm-5")
    try:
        client = get_openai_client()
        speeches_text = "\n".join(f"  [{name}]: {text}" for name, text in round_speeches.items())
        is_last = round_num >= total_rounds
        next_hint = (
            "这是最后一轮辩论，请判断是否足够收敛。"
            if is_last else
            f"这是第 {round_num} 轮（共 {total_rounds} 轮），请指出下一轮需聚焦的问题。"
        )
        user_message = (
            f"当前议程：{agenda.title}，目标：{agenda.goal}\n\n"
            f"第 {round_num} 轮辩论内容：\n{speeches_text}\n\n"
            f"{next_hint}"
            "请输出：1) 本轮主要分歧点 2) 下轮建议聚焦方向，不超过 100 字。"
        )
        moderator_system = (
            "你是一位严谨的会议主持人，负责批结辩论阶段最终将对话导入共识。"
            "不展开新议题，只小结当前动态并引导聊足。"
        )
        resp = client.chat.completions.create(
            model=model_main,
            messages=[
                {"role": "system", "content": moderator_system},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            top_p=0.9,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text
    except Exception as e:
        print(f"  [主持人小结失败] {type(e).__name__}: {str(e)[:60]}")
        return ""


def generate_cross_agenda_minutes(instance: MeetingInstance) -> CrossAgendaMinutes:
    """阶段 2：专家发言 + Secretary LLM 生成跨议程启发纪要。失败时 fallback 简化逻辑。"""

    # --- 专家发言轮 ---
    print("\n-- 全体专家跨议程观察 --")
    expert_observations: Dict[str, str] = {}
    for expert in instance.config.experts:
        print(f"\n[专家 {expert.role_name}] 跨议程观察：")
        obs = call_expert_cross_agenda(expert, instance)
        print(obs)
        expert_observations[expert.role_name] = obs

    # --- Secretary LLM 生成结构化 JSON ---
    print("\n  [秘书] 正在生成跨议程纪要...", end="", flush=True)
    llm_result = call_secretary_for_cross_minutes(expert_observations, instance)

    if llm_result:
        print(" 完成（LLM）。")
        return CrossAgendaMinutes(
            cross_agenda_conflicts=llm_result.get("cross_agenda_conflicts", []),
            cross_agenda_synergies=llm_result.get("cross_agenda_synergies", []),
            topics_for_final_stage=llm_result.get("topics_for_final_stage", []),
        )

    # Fallback
    print(" 使用简化逻辑。")
    deferred_agendas = [
        aid for aid, st in instance.state.agenda_status.items() if st == "deferred"
    ]
    resolved_agendas = [
        aid for aid, st in instance.state.agenda_status.items() if st == "resolved"
    ]
    conflicts = [{
        "agendas": deferred_agendas,
        "description": f"{len(deferred_agendas)} 个议程尚未达成共识。",
        "severity": "medium",
        "suggested_resolution": "在阶段 3 统一处理。",
    }] if deferred_agendas else []
    synergies = [{
        "agendas": resolved_agendas,
        "description": f"{len(resolved_agendas)} 个议程已达成共识，可协同推进。",
        "action_items": ["将已达成共识的议程纪要作为后续执行的输入。"],
    }] if len(resolved_agendas) > 1 else []
    return CrossAgendaMinutes(
        cross_agenda_conflicts=conflicts,
        cross_agenda_synergies=synergies,
        topics_for_final_stage=([f"处理待定议程：{', '.join(deferred_agendas)}"] if deferred_agendas else []),
    )


def run_stage_2(instance: MeetingInstance) -> None:
    """阶段 2：全体横向启发。"""

    state = instance.state
    state.stage = 2

    print("\n>>>> 阶段 2：全体横向启发 <<<<")

    # 汇总阶段 1 纪要
    print("\n-- 阶段 1 议程纪要汇总 --")
    for agenda in instance.config.agendas:
        if agenda.id in state.agenda_minutes:
            minutes = state.agenda_minutes[agenda.id]
            print(f"\n[议程 {agenda.id}] {minutes.agenda_title} - 状态: {minutes.status}")
            print(f"  结论: {minutes.conclusion}")

    # 专家发言 + Secretary 生成跨议程纪要
    cross_minutes = generate_cross_agenda_minutes(instance)
    state.cross_agenda_minutes = cross_minutes

    print("\n-- 跨议程启发纪要（JSON） --")
    print(json.dumps(cross_minutes.to_dict(), ensure_ascii=False, indent=2))

    # 根据 conflicts 自动生成跨议程待定项
    for conflict in cross_minutes.cross_agenda_conflicts:
        related = conflict.get("agendas", [])
        desc = conflict.get("description", "")
        severity = conflict.get("severity", "medium")
        deferred_id = f"cross-d-{len(state.deferred_items) + 1}"
        state.deferred_items.append(
            DeferredItem(
                id=deferred_id,
                type="cross_agenda",
                related_agendas=related,
                reason=f"[severity={severity}] {desc}",
                source_stage=2,
                resolved=False,
            )
        )

    if cross_minutes.cross_agenda_conflicts:
        print(f"\n[阶段 2] 已生成 {len(cross_minutes.cross_agenda_conflicts)} 个跨议程待定项，写入待定池。")

    user_input = prompt_user(
        "\n[阶段 2] 请确认是否继续到阶段 3：输入 '继续' / 其他任意内容视为补充意见：",
        instance,
    )

    if user_input != "继续":
        print(f"[阶段 2] 收到你的意见：{user_input}")
        state.user_notes.append(f"[阶段 2] {user_input}")
        print("当前版本仅记录意见，默认继续到阶段 3。")

    print("\n[阶段 2] 完成，即将进入阶段 3。")


def run_stage_3(instance: MeetingInstance) -> None:
    """阶段 3：统一处理待定池（简化版）。

    当前版本：
    - 按 manual 推荐的顺序：先处理 cross_agenda 待定项，再处理 single_agenda 待定项；
    - 每个待定项：打印回顾信息 + 用户确认循环（共识 / 继续待定）；
    - 用户选择“共识”后，将该待定项标记为 resolved=True。
    后续可以增加：专家分层发言、主持人引导、秘书生成最终决策纪要。
    """

    state = instance.state
    state.stage = 3

    print("\n>>>> 阶段 3：统一处理待定池 <<<<")

    # 按 manual 推荐顺序：先 cross_agenda 再 single_agenda
    cross_deferred = [item for item in state.deferred_items if item.type == "cross_agenda" and not item.resolved]
    single_deferred = [item for item in state.deferred_items if item.type == "single_agenda" and not item.resolved]

    all_to_process = cross_deferred + single_deferred

    if not all_to_process:
        print("当前没有需要处理的待定项，阶段 3 跳过。")
        return

    print(f"待处理待定项总数：{len(all_to_process)}")
    print(f"  - 跨议程待定项：{len(cross_deferred)}")
    print(f"  - 单议程待定项：{len(single_deferred)}")

    for idx, item in enumerate(all_to_process, start=1):
        print(f"\n--- 处理待定项 {idx}/{len(all_to_process)}: [{item.id}] ---")
        print(f"类型: {item.type}")
        print(f"相关议程: {', '.join(item.related_agendas)}")
        print(f"来源阶段: 阶段 {item.source_stage}")
        print(f"待定原因:\n{item.reason}")

        # Phase 2.2: LLM 辅助 — 展示专家建议解决方案
        print("\n  [阶段 3 LLM 辅助] 正在请专家给出建议解决方案...", flush=True)
        deferred_suggestion = call_expert_deferred_suggestion(item, instance)
        if deferred_suggestion:
            print(f"\n[专家建议]\n{deferred_suggestion}")
        else:
            print("  (建议获取失败，继续由用户判断)")

        # 简化版用户确认循环
        while True:
            user_input = prompt_user(
                f"\n[{item.id}] 请确认该待定项状态：输入 '共识' / '继续待定' / 其他任意内容视为补充意见：",
                instance,
            )

            if user_input == "共识":
                print(f"[{item.id}] 已标记为：共识（resolved=True）")
                item.resolved = True
                # 如果是 single_agenda 类型，同步更新议程状态
                if item.type == "single_agenda" and len(item.related_agendas) == 1:
                    agenda_id = item.related_agendas[0]
                    if agenda_id in state.agenda_status:
                        state.agenda_status[agenda_id] = "resolved"
                        print(f"  同步更新议程 {agenda_id} 状态为 resolved。")
                break

            if user_input == "继续待定":
                print(f"[{item.id}] 保持待定状态（resolved=False），跳过该项。")
                break

            # 用户补充意见
            print(f"[{item.id}] 收到你的补充意见（当前版本仅记录，不再触发额外讨论）：\n{user_input}")
            state.user_notes.append(f"[阶段 3 - {item.id}] {user_input}")
            print("你可以再次输入 '共识' 或 '继续待定' 来结束该项，或继续补充意见。")

    print("\n[阶段 3] 完成，所有待定项已处理。")


def call_secretary_for_final_report(instance: MeetingInstance) -> Optional[Dict[str, Any]]:
    """秘书 LLM 根据全会议内容生成最终总报告 JSON。

    输出必填字段：
    - project_summary: {processing_date, total_agendas, resolved_agendas, deferred_with_reason}
    - agenda_decisions: [{agenda_id, title, decision, key_points, risks}]
    - cross_agenda_resolutions: [{issue, resolution, affected_agendas}]
    - overall_risks: [字符串列表]
    - key_assumptions: [字符串列表]
    - action_items: [{item, owner, deadline}]
    - deferred_items: [{id, type, related_agendas, title, reason, review_trigger}]
    """
    REQUIRED = {
        "project_summary", "agenda_decisions", "cross_agenda_resolutions",
        "overall_risks", "key_assumptions", "action_items", "deferred_items"
    }

    from datetime import date
    today = date.today().isoformat()

    state = instance.state
    total = len(instance.config.agendas)
    resolved = sum(1 for s in state.agenda_status.values() if s == "resolved")
    deferred = total - resolved

    # 构建全文上下文
    agenda_ctx = _build_minutes_context(instance)
    deferred_ctx = json.dumps(
        [{
            "id": item.id,
            "type": item.type,
            "related_agendas": item.related_agendas,
            "resolved": item.resolved,
            "reason": item.reason,
        } for item in state.deferred_items],
        ensure_ascii=False, indent=2
    )
    cross_ctx = ""
    if state.cross_agenda_minutes:
        cross_ctx = json.dumps(state.cross_agenda_minutes.to_dict(), ensure_ascii=False, indent=2)

    system_prompt = (
        "你是会议秘书，负责生成最终总报告。"
        "你的回应必须是且仅是一个合法的 JSON 对象，不包含任何额外文字、代码块标记。"
    )
    user_message = (
        f"会议主题：{instance.config.title}\n"
        f"今日日期：{today}\n"
        f"议程纪要：\n{agenda_ctx}\n\n"
        f"跨议程启发纪要：\n{cross_ctx}\n\n"
        f"待定项列表：\n{deferred_ctx}\n\n"
        "请生成最终总报告 JSON，格式如下：\n"
        '{\n'
        f'  "project_summary": {{"processing_date": "{today}", '
        f'"total_agendas": {total}, "resolved_agendas": {resolved}, '
        f'"deferred_with_reason": {deferred}}},\n'
        '  "agenda_decisions": [{"agenda_id": "...", "title": "...", "decision": "...", "key_points": [...], "risks": [...]}],\n'
        '  "cross_agenda_resolutions": [{"issue": "...", "resolution": "...", "affected_agendas": [...]}],\n'
        '  "overall_risks": [...],\n'
        '  "key_assumptions": [...],\n'
        '  "action_items": [{"item": "...", "owner": "...", "deadline": "..."}],\n'
        '  "deferred_items": [{"id": "...", "type": "...", "related_agendas": [...], "title": "...", "reason": "...", "review_trigger": "..."}]\n'
        '}'
    )

    def _call_and_parse(msg: str) -> Optional[Dict[str, Any]]:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "glm-5"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ],
            temperature=0.15,
            top_p=0.9,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None

    try:
        result = _call_and_parse(user_message)
        if result is None:
            return None
        missing = REQUIRED - set(result.keys())
        if missing:
            print(f"  [秘书] 总报告缺少字段 {missing}，触发重写...")
            result = _call_and_parse(
                user_message + f"\n\n上次缺少必填字段：{missing}，请重新输出完整 JSON。"
            )
            if result is None or (REQUIRED - set(result.keys())):
                return None
        # 类型修正
        for field in ("agenda_decisions", "cross_agenda_resolutions", "overall_risks",
                      "key_assumptions", "action_items", "deferred_items"):
            if not isinstance(result.get(field), list):
                result[field] = []
        return result
    except Exception as e:
        print(f"  [秘书] 总报告 JSON 解析失败 ({type(e).__name__}: {str(e)[:60]})，跳过。")
        return None


def generate_final_report(instance: MeetingInstance) -> None:
    """生成并打印最终总报告。"""
    print("\n>>>> 最终总报告生成 <<<<")
    print("  [秘书] 正在生成最终总报告...", end="", flush=True)

    report = call_secretary_for_final_report(instance)

    if report:
        print(" 完成（LLM）。")
        instance.state.final_report = report
    else:
        # Fallback: 简化拼装
        print(" 使用简化逻辑。")
        from datetime import date
        state = instance.state
        total = len(instance.config.agendas)
        resolved = sum(1 for s in state.agenda_status.values() if s == "resolved")
        report = {
            "project_summary": {
                "processing_date": date.today().isoformat(),
                "total_agendas": total,
                "resolved_agendas": resolved,
                "deferred_with_reason": total - resolved,
            },
            "agenda_decisions": [
                {
                    "agenda_id": aid,
                    "title": minutes.agenda_title,
                    "decision": minutes.conclusion,
                    "key_points": minutes.key_decisions,
                    "risks": minutes.risks,
                }
                for aid, minutes in state.agenda_minutes.items()
            ],
            "cross_agenda_resolutions": [
                {
                    "issue": item.reason,
                    "resolution": "已处理" if item.resolved else "待处理",
                    "affected_agendas": item.related_agendas,
                }
                for item in state.deferred_items
            ],
            "overall_risks": [],
            "key_assumptions": [],
            "action_items": [],
            "deferred_items": [
                {
                    "id": item.id,
                    "type": item.type,
                    "related_agendas": item.related_agendas,
                    "title": item.reason[:50],
                    "reason": item.reason,
                    "review_trigger": "暂未设定",
                }
                for item in state.deferred_items
                if not item.resolved
            ],
        }
        instance.state.final_report = report

    print("\n>>>> 总报告（JSON）<<<<")
    print(json.dumps(instance.state.final_report, ensure_ascii=False, indent=2))


def _print_config_preview(agents_path: str, agendas_path: str) -> None:
    """读取并打印当前 agents.json / agendas.json 的完整配置预览。"""

    sep = "=" * 52
    print(f"\n{sep}")
    print("  会议配置预览")
    print(sep)

    # ---------- agents.json 全局 agent 列表 ----------
    agents_data: Optional[Dict[str, Any]] = None
    agent_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(agents_path):
        try:
            with open(agents_path, "r", encoding="utf-8") as f:
                agents_data = json.load(f)
            agent_map = {a.get("id", ""): a for a in agents_data.get("agents", []) if a.get("id")}
        except Exception as e:
            print(f"[配置] 读取 {agents_path} 出错：{type(e).__name__}: {str(e)[:80]}")
    else:
        print(f"[配置] 未找到 {agents_path}，将使用内置 demo 配置。")

    if agent_map:
        print("\n【全体 Agent】")
        for aid, a in agent_map.items():
            kind = a.get("kind", "expert")
            model = a.get("model", "<未配置>")
            temp = a.get("temperature", "?")
            kind_label = {"moderator": "主持人", "secretary": "秘书", "expert": "专家"}.get(kind, kind)
            print(f"  {aid:<22} {kind_label:<5}  model={model}, temp={temp}")

    # ---------- agendas.json 议程列表 ----------
    agendas_data: Optional[Dict[str, Any]] = None
    if os.path.exists(agendas_path):
        try:
            with open(agendas_path, "r", encoding="utf-8") as f:
                agendas_data = json.load(f)
        except Exception as e:
            print(f"[配置] 读取 {agendas_path} 出错：{type(e).__name__}: {str(e)[:80]}")
    else:
        print(f"[配置] 未找到 {agendas_path}，将使用内置 demo 配置。")

    if agendas_data:
        title = agendas_data.get("title", "多议程 AI 会议")
        print(f"\n【会议主题】{title}")
        agendas_list = agendas_data.get("agendas", [])
        if agendas_list:
            print(f"\n【议程列表（共 {len(agendas_list)} 条）】")
            for idx, a in enumerate(agendas_list, 1):
                aids = a.get("id", "")
                atitle = a.get("title", "")
                agoal = a.get("goal", "")
                rounds = a.get("estimated_rounds", 3)
                priority = a.get("priority", "medium")
                participants = a.get("participants", {}) or {}
                expert_ids = participants.get("experts", [])
                moderator_id = participants.get("moderator", "")
                secretary_id = participants.get("secretary", "")

                print(f"\n  [{idx}] {aids}: {atitle}")
                print(f"       目标     : {agoal}")
                print(f"       优先级   : {priority}   预计轮数: {rounds}")
                # 主持人 & 秘书
                mod_label = f"{moderator_id}" if moderator_id else "（默认）"
                sec_label = f"{secretary_id}" if secretary_id else "（默认）"
                print(f"       主持人   : {mod_label}")
                print(f"       秘书     : {sec_label}")
                # 参与专家：列出 id，若在 agent_map 中能找到则附加 system_prompt 前30字
                if expert_ids:
                    print(f"       参与专家 : {', '.join(expert_ids)}")
                    for eid in expert_ids:
                        prompt = (agent_map.get(eid) or {}).get("system_prompt", "")
                        preview = prompt[:40].replace("\n", " ") + ("…" if len(prompt) > 40 else "")
                        print(f"                  → {eid}: {preview}")
                else:
                    print("       参与专家 : 全体专家")
        else:
            print("[配置] agendas.json 中没有任何议程。")
    else:
        print("[配置] 将使用内置 demo 议程配置。")

    print(f"\n{sep}")


def interactive_config_setup(
    agents_path: str = "agents.json",
    agendas_path: str = "agendas.json",
) -> None:
    """在正式开会前展示配置，支持 Y 开始 / R 刷新 / Q 退出的循环确认。

    用户可以在外部编辑 agents.json / agendas.json 后，
    输入 R 来重新读取并显示最新内容，直到对配置满意后输入 Y 开始会议。
    """

    while True:
        _print_config_preview(agents_path, agendas_path)
        print("  Y = 开始会议   R = 刷新配置   Q = 退出")
        ans = input("请输入指令> ").strip().upper()
        if ans == "Y":
            print("[配置] 即将开始会议……")
            return
        if ans == "Q":
            print("[配置] 已退出。")
            sys.exit(0)
        if ans == "R":
            print("[配置] 正在重新读取配置……")
            continue
        print("  未识别的指令，请输入 Y / R / Q。")


def run_meeting(instance: MeetingInstance, save_path: Optional[str] = None) -> None:
    """Run a full meeting instance across all three stages.

    Handles MeetingStopException gracefully (saves and exits).
    Exports all outputs when the meeting completes normally.
    """

    if save_path is None:
        save_path = DEFAULT_STATE_PATH

    print("\n================ 会议实例概览 ================")

    print(f"实例 ID   : {instance.id}")
    print(f"会议主题  : {instance.config.title}")
    print(f"当前阶段  : {instance.state.stage}")
    print(f"当前议程  : {instance.state.current_agenda_id}")

    print("\n议程列表：")
    for agenda in instance.config.agendas:
        print(f"  - {agenda.id}: {agenda.title} (目标: {agenda.goal})")

    print("\n专家配置：")
    for exp in instance.config.experts:
        print(
            f"  - {exp.role_name} (model={exp.model}, "
            f"temperature={exp.temperature}, seed={exp.seed})"
        )

    try:
        # 阶段 1
        run_stage_1(instance)

        # 阶段 2
        run_stage_2(instance)

        # 阶段 3
        run_stage_3(instance)

        # 简要汇总
        print("\n>>>> 会议总结 <<<<")
        for agenda in instance.config.agendas:
            status = instance.state.agenda_status.get(agenda.id, "未标记")
            print(f"  - 议程 {agenda.id} ({agenda.title}) 状态: {status}")

        if instance.state.deferred_items:
            print("\n待定项列表：")
            for item in instance.state.deferred_items:
                resolved_status = "✓ 已解决" if item.resolved else "✗ 未解决"
                print(
                    f"  - [{item.id}] type={item.type}, agendas={item.related_agendas}, "
                    f"resolved={item.resolved} {resolved_status}"
                )
        else:
            print("\n当前没有待定项。")

        total_deferred = len(instance.state.deferred_items)
        resolved_count = sum(1 for item in instance.state.deferred_items if item.resolved)
        unresolved_count = total_deferred - resolved_count

        if total_deferred > 0:
            print(f"\n待定项处理统计：")
            print(f"  - 总数: {total_deferred}")
            print(f"  - 已解决: {resolved_count}")
            print(f"  - 未解决: {unresolved_count}")

        if instance.state.agenda_minutes:
            print("\n>>>> 阶段 1 议程纪要（JSON）<<<<")
            for agenda_id, minutes in instance.state.agenda_minutes.items():
                print(f"\n-- 议程 {agenda_id} 纪要 --")
                print(json.dumps(minutes.to_dict(), ensure_ascii=False, indent=2))

        if instance.state.cross_agenda_minutes:
            print("\n>>>> 阶段 2 横向启发纪要（JSON）<<<<")
            print(json.dumps(instance.state.cross_agenda_minutes.to_dict(), ensure_ascii=False, indent=2))

        # 最终总报告
        generate_final_report(instance)

        # 自动导出所有文件（Phase 1.2）
        print("\n>>>> 导出会议输出文件 <<<<")
        export_meeting_outputs(instance)
        print(f"[导出完成] 文件已写入 output/ 目录。")

    except MeetingStopException:
        print("[中止] 会议已停止，进度已保存。")
        print(f"  下次可用 --resume {save_path} 恢复会议。")
        sys.exit(0)


def main() -> None:
    """Entry point.

    Usage:
        uv run python main.py                         # start a new meeting
        uv run python main.py --resume output/demo-001_state.json  # resume
    """
    load_environment()

    resume_path: Optional[str] = None
    args = sys.argv[1:]
    if "--resume" in args:
        idx = args.index("--resume")
        if idx + 1 < len(args):
            resume_path = args[idx + 1]
        else:
            print("[错误] --resume 需要指定快照文件路径，例如: --resume output/demo-001_state.json")
            sys.exit(1)

    if resume_path:
        if not os.path.exists(resume_path):
            print(f"[错误] 指定的快照文件不存在：{resume_path}")
            sys.exit(1)
        print(f"[恢复会议] 正在从 {resume_path} 加载中...")
        instance = load_meeting_instance(resume_path)
        print(f"[恢复会议] 已加载实例 {instance.id}")
    else:
        # 开会前展示配置并让用户确认
        interactive_config_setup()
        instance = create_demo_instance()

    run_meeting(instance, save_path=resume_path or DEFAULT_STATE_PATH)


if __name__ == "__main__":
    main()
