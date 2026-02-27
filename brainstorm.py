"""
简化版多专家头脑风暴系统
基于AutoGen，支持人工实时介入控场
使用DMXAPI/LiteLLM + AutoGen 0.4+

流程：
1. 独立立论阶段：每位专家独立发表观点（彼此不可见）
2. 主持人总结：梳理观点异同、识别分歧点
3. 头脑风暴阶段：专家互相可见，深入讨论
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 加载环境变量
load_dotenv()


def create_model_client(model_name: str = None, temperature: float = 0.8):
    """创建模型客户端（支持DMXAPI/LiteLLM）"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE", "https://www.dmxapi.cn/v1")
    model = model_name or os.getenv("MODEL_NAME", "glm-5")
    
    model_info = {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "family": "unknown",
    }
    
    return OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        model_info=model_info,
    )


def load_agents_config(config_file: str = "agents1.json"):
    """从配置文件加载专家配置"""
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("agents", [])


def review_arrangements(
    base_host: Optional[AssistantAgent],
    base_experts: List[AssistantAgent],
    filename: str = "arrangements.json",
) -> Tuple[Optional[AssistantAgent], List[AssistantAgent]]:
    """在会议正式开始前，按 arrangements.json 反复确认本轮将采用的秘书与专家。

    每次循环：
    - 读取并应用 arrangements.json（由 apply_arrangements 负责解析与容错）
    - 在终端公布当前计算出的秘书与专家名单
    - 用户输入非空内容视为刚修改过文件，将再次读取并公布；
      只有在用户直接回车时才最终确认本轮安排并返回。
    """
    current_host = base_host
    current_experts = base_experts

    while True:
        print("\n" + "-" * 60)
        print(f"📂 正在根据 {filename} 计算本轮将采用的秘书与专家...")
        current_host, current_experts = apply_arrangements(base_host, base_experts, filename)

        user_input = input(
            "\n如已确认以上安排，直接回车开始会议；\n"
            "如果你刚刚修改过 arrangements.json，请输入任意内容后回车以重新载入："
        ).strip()
        if not user_input:
            break

    return current_host, current_experts


def apply_arrangements(
    host_agent: Optional[AssistantAgent],
    expert_agents: List[AssistantAgent],
    filename: str = "arrangements.json",
) -> Tuple[Optional[AssistantAgent], List[AssistantAgent]]:
    """根据 arrangements.json 选择本次会议实际使用的秘书与专家集合。

    - 优先查找 id="default" 的议程；否则取列表中的第一个议程。
    - secretary 字段指定本次会议使用的秘书 name。
    - experts 列表指定本次会议参加的专家 name 集合。
    未命中时回退到 agents1.json 中默认加载的角色。
    """
    if not os.path.exists(filename):
        return host_agent, expert_agents

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return host_agent, expert_agents

    agendas = data.get("agendas") or []
    agenda = None
    for item in agendas:
        if item.get("id") == "default":
            agenda = item
            break
    if agenda is None and agendas:
        agenda = agendas[0]
    if agenda is None:
        print("⚠️ arrangements.json 中未找到任何有效议程配置，将按 agents1.json 的默认角色配置运行。")
        return host_agent, expert_agents

    secretary_name = agenda.get("secretary")
    expert_names = agenda.get("experts") or []

    # 构建 name -> Agent 映射，方便通过名字选人
    name_to_agent: dict[str, AssistantAgent] = {}
    if host_agent:
        name_to_agent[host_agent.name] = host_agent
    for agent in expert_agents:
        if agent.name not in name_to_agent:
            name_to_agent[agent.name] = agent

    # 按 arrangements.json 重新确定秘书
    new_host = host_agent
    if secretary_name:
        if secretary_name in name_to_agent:
            new_host = name_to_agent[secretary_name]
        else:
            print(f"⚠️ arrangements.json 中 secretary=‘{secretary_name}’ 在 agents1.json 中找不到，对应秘书保持为 {host_agent.name if host_agent else '（无）'}。")

    # 按 arrangements.json 过滤/排序专家列表
    new_experts: List[AssistantAgent] = []
    if expert_names:
        for name in expert_names:
            agent = name_to_agent.get(name)
            if not agent:
                print(f"⚠️ arrangements.json 中 experts 项 ‘{name}’ 在 agents1.json 中找不到，将忽略该专家。")
                continue
            if agent is not new_host and agent not in new_experts:
                new_experts.append(agent)
    else:
        # 未显式指定 experts 时，沿用原有专家列表，但排除秘书自身
        for agent in expert_agents:
            if agent is not new_host:
                new_experts.append(agent)

    # 公布本轮最终将采用的角色安排
    print("\n🧾 本轮将采用的角色安排（来自 arrangements.json）：")
    agenda_title = agenda.get("title") or agenda.get("id") or "(未命名议程)"
    print(f"  议程: {agenda_title}")
    print(f"  秘书: {new_host.name if new_host else '（无）'}")
    if new_experts:
        print("  专家:")
        for idx, ag in enumerate(new_experts, start=1):
            print(f"    {idx}. {ag.name}")
    else:
        print("  专家: （无专家参与）")

    return new_host, new_experts


def create_agents(config_file: str = "agents1.json"):
    """创建所有Agent（秘书+专家），支持不同模型"""
    agents_config = load_agents_config(config_file)
    host_agent = None
    expert_agents = []
    
    for agent_config in agents_config:
        agent = AssistantAgent(
            name=agent_config["name"],
            model_client=create_model_client(
                model_name=agent_config.get("model"),
                temperature=agent_config.get("temperature", 0.8)
            ),
            system_message=agent_config["system_prompt"]
        )
        
        if agent_config.get("kind") == "secretary":
            host_agent = agent
            print(f"  ✅ 秘书 {agent_config['name']} ({agent_config.get('model', 'glm-5')})")
        else:
            expert_agents.append(agent)
            print(f"  ✅ 专家 {agent_config['name']} ({agent_config.get('model', 'glm-5')})")
    
    return host_agent, expert_agents


def save_chat_history(messages, filename="brainstorm_history.json"):
    """保存对话历史到文件"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 对话历史已保存到：{filename}")
def save_announcement(
    session_chat: List[dict],
    draft: str,
    stage_label: str,
    published: bool,
    decision_input: str,
    filename: str = "secretaryannouncements.json",
):
    """追加一条秘书公示记录到 secretaryannouncements.json"""
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            records = []
    except Exception:
        records = []

    record = {
        "id": len(records) + 1,
        "stage": stage_label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chat": session_chat,
        "announcement": draft,
        "decision_input": decision_input,
        "published": published,
    }
    records.append(record)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def open_draft_session(stage_label: str, filename: str = "secretaryannouncements.json") -> int:
    """在 secretaryannouncements.json 中为本次私聊开屏一条新记录，返回该记录的 id。

    记录初始状态: announcement 为空，published = False。
    后续由 update_draft_in_place 和 finalize_draft 逐步完善。
    """
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            records = []
    except Exception:
        records = []

    new_id = len(records) + 1
    record = {
        "id": new_id,
        "stage": stage_label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chat": [],
        "announcement": "",
        "decision_input": "",
        "published": False,
    }
    records.append(record)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return new_id


def update_draft_in_place(
    record_id: int,
    session_chat: List[dict],
    new_text: str,
    filename: str = "secretaryannouncements.json",
):
    """静默将指定 id 的记录的 announcement 字段原地更新。

    终端不打印任何内容，就像文本编辑器静默修改文件一样。
    """
    try:
        if not os.path.exists(filename):
            return
        with open(filename, "r", encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            if rec["id"] == record_id:
                rec["chat"] = session_chat
                rec["announcement"] = new_text
                break
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def finalize_draft(
    record_id: int,
    session_chat: List[dict],
    decision_input: str,
    published: bool,
    filename: str = "secretaryannouncements.json",
):
    """将指定 id 的记录标记为已决策（发表/不发表）。终端不打印。"""
    try:
        if not os.path.exists(filename):
            return
        with open(filename, "r", encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            if rec["id"] == record_id:
                rec["chat"] = session_chat
                rec["decision_input"] = decision_input
                rec["published"] = published
                break
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


async def maybe_update_announcement(
    host: AssistantAgent,
    topic: str,
    stage_label: str,
    current_draft: Optional[str],
    last_user_msg: str,
    last_secretary_reply: str,
    stage_code: str,
    all_messages: List[dict],
) -> Tuple[bool, Optional[str]]:
    """隐形意图识别：判断主持人这一轮说的话是否需要更新 announcement。

    如果需要，返回 (True, 新全文)；否则返回 (False, None)。
    该函数在终端不打印任何内容，完全静默运行。
    """
    current_draft_text = current_draft or "（尚未起草）"

    system_prompt = (
        f"你是会议秘书，正在帮主持人起草一份公告（announcement）草稿。\n"
        f"主题: 「{topic}」，当前阶段: 「{stage_label}」。\n\n"
        f"【当前公告草稿】\n{current_draft_text}\n\n"
        f"【最新一轮私聊】\n"
        f"主持人: {last_user_msg}\n"
        f"秘书: {last_secretary_reply}\n\n"
        "请判断：主持人这一轮的发言，是否包含对公告内容的修改意图？\n"
        "（例如：提出新的重点、要求改语气/措辞、要删某个点、根据专家发言做调整）\n"
        "如果没有（只是闲聊、问解释、表达情绪），则 should_update = false。\n\n"
        "只输出一个 JSON 对象，格式如下，不要包含任何其他内容：\n"
        '{"should_update": true, "new_announcement": "改好后的公告全文"}\n'
        '{"should_update": false, "new_announcement": ""}'
    )

    messages = [TextMessage(content=system_prompt, source="system")]
    if all_messages:
        for msg in all_messages[-15:]:
            src = msg.get("source", "Other")
            messages.append(TextMessage(
                content=f"[{src}] {msg.get('content', '')}",
                source=src,
            ))

    try:
        response = await host.on_messages(messages, cancellation_token=None)
        raw = response.chat_message.content.strip()
        # 容迍模型用 ```json ... ``` 包裹返回
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        should_update = bool(data.get("should_update", False))
        new_text = data.get("new_announcement", "").strip() if should_update else None
        return should_update, new_text if new_text else None
    except Exception:
        return False, None


def append_secretary_log(line: str, filename: str = "secretarynote.txt"):
    """追加秘书笔记到本地文件"""
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # 日志失败不影响主流程
        pass


async def _secretary_chat_reply(
    host: AssistantAgent,
    session_chat: List[dict],
    topic: str,
    stage_label: str,
    meeting_history: Optional[List[dict]] = None,
) -> str:
    """秘书在私聊模式下的自然对话回复。

    会参考本次会议的公开历史记录（专家发言、主持人公开发言等）
    再结合本次私聊的对话内容，像真人秘书一样用口语方式帮主持人
    理解刚才大家在说什么，以及一起推演等一下可能要对专家公开说什么。

    注意：这里只是私下沟通，不要直接写正式的 announcement 或结构化总结。
    """
    messages = [
        TextMessage(
            content=(
                f"你正在和主持人私下聊，主题是「{topic}」，当前为「{stage_label}」阶段。\n"
                "你可以参考下面这场会议中，主持人和各位专家之前的公开发言记录，"
                "帮助主持人用口语方式理解刚才大家在说什么，以及一起商量等一下要对专家公开说什么。\n"
                "在私聊阶段：\n"
                "1）请用轻松自然的语气回答主持人的问题，像真人秘书小声解释一样；\n"
                "2）可以引用或概括专家观点，但不要输出正式的公示稿或结构化总结；\n"
                "3）不要替主持人直接写 announcement，只是帮他理思路、提建议。"
            ),
            source="system",
        )
    ]

    # 先喂入会议公开历史（只取最近若干条，避免上下文过长）
    if meeting_history:
        recent_history = meeting_history[-30:]
        for msg in recent_history:
            src = msg.get("source", "Other")
            content = msg.get("content", "")
            wrapped = f"[{src}] {content}"
            messages.append(TextMessage(content=wrapped, source=src))

    # 再追加本次私聊的回合记录
    for msg in session_chat:
        src = "Host" if msg["role"] == "user" else host.name
        messages.append(TextMessage(content=msg["content"], source=src))

    response = await host.on_messages(messages, cancellation_token=None)
    return response.chat_message.content


async def secretary_chat_session(
    host_agent: AssistantAgent,
    all_messages: List[dict],
    secretary_notes: List[dict],
    topic: str,
    stage_label: str,
    stage_code: str,
) -> Optional[str]:
    """与秘书进行多轮私聊对话，背后静默向 secretaryannouncements.json 实时更新公告草稿。

    流程：
    - 每轮私聊: 秘书口语回应 → maybe_update_announcement 隐形判断 → 如有修改就静默写入 JSON
    - 终端只显示秘书的口语回应 + 公告是否被更新的提示符
    - 回车结束私聊后展示最终草稿，询问发表/不发表
    """
    print(f"\n📝 秘书凑过来（{stage_label}）... 📏 公告草稿实时在 secretaryannouncements.json 更新")
    session_chat: List[dict] = []
    current_draft: Optional[str] = None

    # 就地开一个插槽，返回 record_id以便后续就地更新
    record_id = open_draft_session(stage_label)

    while True:
        user_msg = input("《私语》你对秘书说（直接回车结束私聊）：").strip()
        if not user_msg:
            break

        session_chat.append({"role": "user", "content": user_msg})
        secretary_notes.append({"source": "HostPrivate", "content": f"[{stage_label}] {user_msg}"})
        append_secretary_log(f"[PRIVATE][{stage_label}] {user_msg}")

        # 秘书口语回应（看会议历史 + 本次私聊，自然地趟你聊）
        reply = await _secretary_chat_reply(host_agent, session_chat, topic, stage_label, all_messages)
        print(f"\n🤫 [秘书]: {reply}\n")
        session_chat.append({"role": "secretary", "content": reply})
        secretary_notes.append({"source": "SecretaryPrivate", "content": reply})
        append_secretary_log(f"[PRIVATE_REPLY][{stage_label}] {reply}")

        # 隐形意图识别：判断是否需要更新 announcement
        should_update, new_text = await maybe_update_announcement(
            host_agent,
            topic,
            stage_label,
            current_draft,
            user_msg,
            reply,
            stage_code,
            all_messages,
        )
        if should_update and new_text:
            current_draft = new_text
            update_draft_in_place(record_id, session_chat, current_draft)
            print("📏 announcement 已更新（见 secretaryannouncements.json）")

    # 私聊结束：如果一轮都没有触发 maybe_update，做一次兼底全量生成
    if current_draft is None:
        print(f"\n⏳ 秘书正在起草公示稿...")
        ctx_draft = all_messages + secretary_notes
        current_draft = await get_host_response(host_agent, ctx_draft, topic, stage=stage_code)
        update_draft_in_place(record_id, session_chat, current_draft)

    print(f"\n{'=' * 60}")
    print(f"📄 公示稿内容（{stage_label}）")
    print(f"{'=' * 60}")
    print(current_draft)
    print(f"{'=' * 60}\n")

    decision_raw = input("发表此次 announcement？(y=发表 / n=不发表)：").strip().lower()
    published = decision_raw in ["", "y", "yes", "是", "ok"]

    finalize_draft(record_id, session_chat, decision_raw, published)

    if published:
        append_secretary_log(f"[PUBLIC][{stage_label}][秘书公示]: {current_draft}")
        return current_draft
    else:
        print("ℹ️ 已取消发表，公示稿不对外公布。")
        return None


async def get_expert_response(expert: AssistantAgent, context: List[dict]) -> str:
    """获取单个专家的回复"""
    messages = []
    for msg in context:
        messages.append(TextMessage(content=msg["content"], source=msg["source"]))
    
    response = await expert.on_messages(messages, cancellation_token=None)
    return response.chat_message.content


async def get_host_response(host: AssistantAgent, context: List[dict], topic: str, stage: str = "general") -> str:
    """获取秘书回复

    stage:
        'chat'              -- 纯对话模式，秘书自然回应你说的话，不强制产出结构化总结
        'opening_summary'   -- 拟一份正式的立论总结公示稿
        'brainstorm_comment'-- 拟一份正式的头脑风暴点评公示稿
    """
    messages = []
    for msg in context:
        messages.append(TextMessage(content=msg["content"], source=msg["source"]))

    if stage == "opening_statement":
        guide = (
            f"请根据主持人刚才的意图，以秘书身份起草一份开场评论 / 定调发言公示稿，"
            f"供主持人审阅后向全体专家发布。内容应符合主持人地说明本次会议的目的、期望和调性。"
        )
        messages.append(TextMessage(content=guide, source="system"))
    elif stage == "opening_summary":
        guide = (
            f"请根据以上内容，以秘书身份为「{topic}」的立论阶段起草一份完整的正式公示稿，"
            f"供主持人审阅后向全体专家发布。包含：共识点、主要分歧、待探讨问题。"
        )
        messages.append(TextMessage(content=guide, source="system"))
    elif stage == "brainstorm_comment":
        guide = (
            f"请根据以上内容，以秘书身份为「{topic}」本轮头脑风暴起草一份正式公示稿，"
            f"供主持人审阅后向全体专家发布。指出关键问题、引导方向或需要补充的点。"
        )
        messages.append(TextMessage(content=guide, source="system"))
    # stage == 'chat'：纯对话，不追加任何 guide，让秘书自然回应

    response = await host.on_messages(messages, cancellation_token=None)
    return response.chat_message.content


def parse_user_choice(user_input: str) -> Tuple[bool, bool, bool]:
    """解析用户选择

    Returns:
        (should_continue, has_1, has_3):
        是否继续, 是否包含操作1, 是否包含操作3
    """
    if not user_input:
        # 空输入等价于选择2：继续
        return True, False, False

    user_input = user_input.lower().strip()

    # 检查是否结束
    if user_input in ["结束", "end", "stop", "quit", "exit", "q"]:
        return False, False, False

    has_1 = "1" in user_input
    has_3 = "3" in user_input

    # 只有2表示纯继续
    if user_input == "2":
        return True, False, False

    return True, has_1, has_3


def print_menu(is_after_statement: bool = False):
    """打印操作菜单"""
    print(f"\n{'─' * 60}")
    if is_after_statement:
        print("💡 请选择操作（可多选，如输入'13'表示先插话再总结）：")
    else:
        print("💡 请选择操作（可多选）：")
    print("   1. 输入你的想法 → 插话补充（可在数字后输入内容）")
    print("   2. 直接回车 → 跳过，继续下一位")
    print("   3. 让秘书拟总结/点评 → 先与秘书私语，再决定是否对外发布")
    print("   结束 → 终止整个会议")
    print(f"{'─' * 60}")


async def run_brainstorm(topic: str, max_rounds: int = 10):
    """
    运行头脑风暴会议 - 两阶段流程
    
    Args:
        topic: 讨论主题
        max_rounds: 最大讨论轮数（头脑风暴阶段）
    """
    print("=" * 60)
    print("🚀 多专家头脑风暴会议已启动")
    print("=" * 60)
    print(f"\n📋 讨论主题：{topic}")
    print(f"🎯 头脑风暴轮数：{max_rounds} 轮")
    print("\n💡 系统配置：")
    print(f"   • API: {os.getenv('OPENAI_API_BASE', 'https://www.dmxapi.cn/v1')}")
    print("=" * 60)
    
    # 1. 创建所有Agent（主持人+专家）
    print("\n🤖 正在初始化Agent...")
    host_agent, expert_agents = create_agents()

    # 1.1 在正式开会前，根据 arrangements.json 反复确认本轮将采用的秘书与专家
    host_agent, expert_agents = review_arrangements(host_agent, expert_agents)
    expert_names = [agent.name for agent in expert_agents]

    # 此处不再重复打印角色，review_arrangements/apply_arrangements 已经公布了本轮安排
    
    # 2. 初始化对话历史
    all_messages = []
    secretary_notes: List[dict] = []  # 仅秘书可见的私语笔记

    # 2.1 开始前的秘书面板
    while True:
        print_menu(is_after_statement=True)
        user_input = input("你的选择：").strip()
        should_continue, has_1, has_3 = parse_user_choice(user_input)

        if not should_continue:
            print("\n🛑 会议被你终止")
            save_chat_history(all_messages)
            return all_messages

        # 1. 你对所有人说
        if has_1:
            public_msg = input("请输入你要对所有专家公开说的话（直接回车表示不说）：").strip()
            if public_msg:
                all_messages.append({"source": "Host", "content": public_msg})
                print(f"\n🎤 [你对大家说]: {public_msg}")
                append_secretary_log(f"[PUBLIC][开场][你]: {public_msg}")

        # 3. 和秘书私语，让秘书拟一段开场总结
        if has_3 and host_agent:
            print("\n📝 秘书凑过来，等你私下交代这次会议要怎么定调...")
            announced = await secretary_chat_session(
                host_agent, all_messages, secretary_notes, topic,
                stage_label="开场", stage_code="opening_statement"
            )
            if announced:
                all_messages.append({"source": host_agent.name, "content": announced})

        # 2. 选择2：跳过，退出当前面板，进入独立立论
        if user_input == "2":
            break
    # ==================== 阶段一：独立立论 ====================
    print("\n" + "=" * 60)
    print("📢 阶段一：独立立论")
    print("=" * 60)
    print(f"\n📝 讨论主题：{topic}")
    print("每位专家将独立发表初始观点（彼此不可见）\n")
    
    # 收集每位专家的立论
    opening_statements = []
    
    for expert in expert_agents:
        print(f"\n⏳ {expert.name} 正在立论...")
        
        # 为每个专家生成独立的立论提示
        initial_prompt = f"""请围绕「{topic}」提供一份完整的{expert.name}专业方案/分析。

这是立论阶段，你需要：
1. 从{expert.name}的专业视角，系统性地分析问题
2. 提供具体、可落地的方案或建议（不是简单观点），以及为什么
3. 包含：现状分析、核心思路、具体措施、预期效果
4. 篇幅适中，确保内容完整、有深度
5. 有附录，用于解释一些不常见或可能有歧义的(如果没有就不写)，“(在我的语境下)是什么”，“为什么”，方便沟通

请像给客户提供正式咨询报告一样，给出专业、详实的方案。"""
        
        try:
            # 独立立论：只给初始话题，看不到其他人的观点
            context = [{"source": "Host", "content": initial_prompt}]
            response = await get_expert_response(expert, context)
            
            print(f"\n💬 [{expert.name}]: {response}\n")
            opening_statements.append({"source": expert.name, "content": response})
            
        except Exception as e:
            print(f"\n❌ {expert.name} 立论出错: {e}")
            continue
    
    # 立论完成后，秘书面板
    while True:
        print_menu(is_after_statement=True)
        user_input = input("你的选择：").strip()
        should_continue, has_1, has_3 = parse_user_choice(user_input)

        if not should_continue:
            print("\n🛑 会议被你终止")
            save_chat_history(all_messages)
            return all_messages
        
        # 1. 你对所有人说
        if has_1:
            public_msg = input("请输入你要对所有专家公开说的话（直接回车表示不说）：").strip()
            if public_msg:
                all_messages.append({"source": "Host", "content": public_msg})
                print(f"\n🎤 [你对大家说]: {public_msg}")
                append_secretary_log(f"[PUBLIC][立论后][你]: {public_msg}")
        
        # 3. 秘书拟总结
        if has_3 and host_agent:
            announced = await secretary_chat_session(
                host_agent, all_messages, secretary_notes, topic,
                stage_label="立论后", stage_code="opening_summary"
            )
            if announced:
                all_messages.append({"source": host_agent.name, "content": announced})
        
        # 2. 选择2：跳过，退出当前面板，进入头脑风暴
        if user_input == "2":
            break
    
    # ==================== 阶段二：头脑风暴 ====================
    print("\n" + "=" * 60)
    print("🧠 阶段二：头脑风暴")
    print("=" * 60)
    print("专家现在可以看到彼此的观点，开始深入讨论\n")
    
    # 添加阶段转换提示
    transition_msg = """立论阶段结束，现在进入头脑风暴阶段。

各位专家已经看到了彼此的完整方案。请基于其他专家的方案进行：
1. 补充：在自己专业领域内，补充其他专家方案中缺失的内容
2. 交叉验证：指出自己专业视角下，其他方案的可行性、风险或改进建议
3. 整合建议：提出如何协调不同专家方案，形成更完善的综合方案

请保持专业、建设性的态度，目标是形成一份融合各方专长的完整解决方案。"""
    all_messages.append({"source": "Host", "content": transition_msg})
    
    round_num = 1
    
    while round_num <= max_rounds:
        print(f"\n{'=' * 60}")
        print(f"🔄 第 {round_num} 轮讨论")
        print(f"{'=' * 60}")
        
        for expert in expert_agents:
            print(f"\n⏳ {expert.name} 正在思考...")
            
            try:
                # 头脑风暴：可以看到所有历史（包括立论和之前的发言）
                response = await get_expert_response(expert, all_messages)
                
                print(f"\n💬 [{expert.name}]: {response}\n")
                all_messages.append({"source": expert.name, "content": response})
                
                # 专家发言后，主持人介入
                while True:
                    print_menu(is_after_statement=False)
                    user_input = input("你的选择：").strip()
                    
                    should_continue, has_1, has_3 = parse_user_choice(user_input)
                    
                    if not should_continue:
                        print("\n🛑 会议被主持人终止")
                        save_chat_history(all_messages)
                        return all_messages
                    
                    # 处理用户插话（1）
                    if has_1:
                        public_msg = input("请输入你要对所有专家公开说的话（直接回车表示不说）：").strip()
                        if public_msg:
                            all_messages.append({"source": "Host", "content": public_msg})
                            print(f"\n🎤 [你对大家说]: {public_msg}")
                            append_secretary_log(f"[PUBLIC][第{round_num}轮][你]: {public_msg}")
                    
                    # 处理秘书总结/点评（3）
                    if has_3 and host_agent:
                        announced = await secretary_chat_session(
                            host_agent, all_messages, secretary_notes, topic,
                            stage_label=f"第{round_num}轮", stage_code="brainstorm_comment"
                        )
                        if announced:
                            all_messages.append({"source": host_agent.name, "content": announced})
                        
                        # 公示后继续显示菜单
                        continue
                    
                    # 继续下一位专家
                    break
                    
            except Exception as e:
                print(f"\n❌ {expert.name} 发言出错: {e}")
                continue
        
        round_num += 1
        
        # 检查是否继续下一轮
        if round_num <= max_rounds:
            print(f"\n{'─' * 60}")
            cont = input(f"第 {round_num-1} 轮结束，是否继续第 {round_num} 轮？(回车继续/输入'结束'停止): ").strip()
            if cont.lower() in ["结束", "end", "stop", "n", "no", "q"]:
                print("\n🛑 会议被主持人终止")
                break
    
    # 保存对话历史
    save_chat_history(all_messages)
    
    print("\n" + "=" * 60)
    print("🎉 头脑风暴会议已结束")
    print("=" * 60)
    
    return all_messages


def reset_meeting_logs():
    """在每次会议实例开始前刷新关键日志文件。

    为避免不同会议实例之间的笔记和公示记录相互混杂，
    启动新会议前删除上一轮的 secretarynote / secretaryannouncements / brainstorm_history。
    """
    log_files = [
        "secretarynote.txt",
        "secretaryannouncements.json",
        "brainstorm_history.json",
    ]
    for filename in log_files:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception:
            # 刷新失败不影响主流程
            pass


def main():
    """主入口"""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误：未设置 OPENAI_API_KEY 环境变量")
        print("请检查 .env 文件或设置环境变量")
        return

    # 每次新的会议实例开始前刷新日志文件，避免不同实例之间相互污染
    reset_meeting_logs()
    
    print("\n🧠 AutoGen 多专家头脑风暴系统")
    print("-" * 40)
    
    default_topic = "AI驱动的个人知识管理工具"
    user_input = input(f"\n请输入讨论主题（直接回车使用默认主题：{default_topic}）：").strip()
    topic = user_input if user_input else default_topic
    
    rounds_input = input("请输入头脑风暴轮数（直接回车默认3轮）：").strip()
    max_rounds = int(rounds_input) if rounds_input.isdigit() else 3
    
    asyncio.run(run_brainstorm(topic, max_rounds))


if __name__ == "__main__":
    main()
