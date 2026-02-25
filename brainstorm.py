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
from typing import List, Tuple
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


def create_agents(config_file: str = "agents1.json"):
    """创建所有Agent（主持人+专家），支持不同模型"""
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
        
        if agent_config.get("kind") == "host":
            host_agent = agent
            print(f"  ✅ 主持人 {agent_config['name']} ({agent_config.get('model', 'glm-5')})")
        else:
            expert_agents.append(agent)
            print(f"  ✅ 专家 {agent_config['name']} ({agent_config.get('model', 'glm-5')})")
    
    return host_agent, expert_agents


def save_chat_history(messages, filename="brainstorm_history.json"):
    """保存对话历史到文件"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 对话历史已保存到：{filename}")


async def get_expert_response(expert: AssistantAgent, context: List[dict]) -> str:
    """获取单个专家的回复"""
    messages = []
    for msg in context:
        messages.append(TextMessage(content=msg["content"], source=msg["source"]))
    
    response = await expert.on_messages(messages, cancellation_token=None)
    return response.chat_message.content


async def get_host_response(host: AssistantAgent, context: List[dict], topic: str, stage: str = "general") -> str:
    """获取主持人回复"""
    messages = []
    for msg in context:
        messages.append(TextMessage(content=msg["content"], source=msg["source"]))
    
    # 根据阶段添加引导提示
    if stage == "opening_summary":
        guide = f"请作为主持人，对各位专家关于「{topic}」的立论进行总结。梳理共识点、分歧点和待探讨问题。"
        messages.append(TextMessage(content=guide, source="system"))
    elif stage == "brainstorm_comment":
        guide = f"请作为主持人，针对刚才的讨论进行点评。指出关键问题、引导方向或提出需要补充的点。"
        messages.append(TextMessage(content=guide, source="system"))
    
    response = await host.on_messages(messages, cancellation_token=None)
    return response.chat_message.content


def parse_user_choice(user_input: str) -> Tuple[bool, bool, str]:
    """
    解析用户选择
    
    Returns:
        (should_continue, host_summary, user_comment): 
        是否继续, 是否需要主持人总结, 用户插话内容
    """
    if not user_input:
        return True, False, ""
    
    user_input = user_input.lower().strip()
    
    # 检查是否结束
    if user_input in ["结束", "end", "stop", "quit", "exit", "q"]:
        return False, False, ""
    
    # 解析多选（如 "13", "31", "1", "3"）
    has_1 = "1" in user_input
    has_3 = "3" in user_input
    
    # 如果只有数字2，表示跳过
    if user_input == "2":
        return True, False, ""
    
    # 提取用户评论（去掉数字后的内容）
    comment = user_input
    for char in "123":
        comment = comment.replace(char, "")
    comment = comment.strip()
    
    return True, has_3, comment if has_1 else ""


def print_menu(is_after_statement: bool = False):
    """打印操作菜单"""
    print(f"\n{'─' * 60}")
    if is_after_statement:
        print("💡 请选择操作（可多选，如输入'13'表示先插话再总结）：")
    else:
        print("💡 请选择操作（可多选）：")
    print("   1. 输入你的想法 → 插话补充（可在数字后输入内容）")
    print("   2. 直接回车 → 跳过，继续下一位")
    print("   3. 主持人总结 → 梳理观点异同、识别分歧点")
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
    expert_names = [agent.name for agent in expert_agents]
    
    if host_agent:
        print(f"\n🎤 主持人：{host_agent.name}")
    print(f"👥 参会专家：{', '.join(expert_names)}")
    
    # 2. 初始化对话历史
    all_messages = []
    
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
    
    # 立论完成后，主持人介入
    print(f"\n{'=' * 60}")
    print("🎯 立论阶段完成，主持人请介入")
    print(f"{'=' * 60}")
    
    # 添加所有立论到历史
    all_messages.extend(opening_statements)
    
    # 主持人选择操作
    while True:
        print_menu(is_after_statement=True)
        user_input = input("你的选择：").strip()
        
        should_continue, need_summary, user_comment = parse_user_choice(user_input)
        
        if not should_continue:
            print("\n🛑 会议被主持人终止")
            save_chat_history(all_messages)
            return all_messages
        
        # 处理用户插话
        if user_comment:
            all_messages.append({"source": "Host", "content": user_comment})
            print(f"\n🎤 [主持人插话]: {user_comment}")
        
        # 处理主持人总结
        if need_summary and host_agent:
            print("\n📝 主持人正在梳理观点...")
            summary = await get_host_response(host_agent, all_messages, topic, stage="opening_summary")
            
            print(f"\n{'=' * 60}")
            print("📊 主持人总结")
            print(f"{'=' * 60}")
            print(summary)
            print(f"{'=' * 60}\n")
            
            all_messages.append({"source": host_agent.name, "content": summary})
        
        # 如果用户没有选3（总结），或者已经总结完了，进入下一阶段
        if not need_summary or user_input:
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
                    
                    should_continue, need_summary, user_comment = parse_user_choice(user_input)
                    
                    if not should_continue:
                        print("\n🛑 会议被主持人终止")
                        save_chat_history(all_messages)
                        return all_messages
                    
                    # 处理用户插话
                    if user_comment:
                        all_messages.append({"source": "Host", "content": user_comment})
                        print(f"\n🎤 [主持人插话]: {user_comment}")
                    
                    # 处理主持人总结
                    if need_summary and host_agent:
                        print("\n📝 主持人正在梳理本轮观点...")
                        summary = await get_host_response(host_agent, all_messages, topic, stage="brainstorm_comment")
                        
                        print(f"\n{'=' * 60}")
                        print(f"📊 主持人点评（第{round_num}轮）")
                        print(f"{'=' * 60}")
                        print(summary)
                        print(f"{'=' * 60}\n")
                        
                        all_messages.append({"source": host_agent.name, "content": summary})
                        
                        # 总结后继续显示菜单，让用户选择是否继续
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


def main():
    """主入口"""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误：未设置 OPENAI_API_KEY 环境变量")
        print("请检查 .env 文件或设置环境变量")
        return
    
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
