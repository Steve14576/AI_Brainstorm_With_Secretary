"""
简化版多专家头脑风暴系统
基于AutoGen，支持人工实时介入控场
使用DMXAPI/LiteLLM + AutoGen 0.4+
"""

import os
import json
import asyncio
from typing import List
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
    
    # 为非标准模型提供 model_info
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


def create_expert_agents(config_file: str = "agents1.json"):
    """创建各领域专家Agent，支持不同模型"""
    agents_config = load_agents_config(config_file)
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
        expert_agents.append(agent)
        print(f"  ✅ {agent_config['name']} ({agent_config.get('model', 'glm-5')})")
    
    return expert_agents


def save_chat_history(messages, filename="brainstorm_history.json"):
    """保存对话历史到文件"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 对话历史已保存到：{filename}")


async def get_expert_response(expert: AssistantAgent, context: List[dict]) -> str:
    """获取单个专家的回复"""
    # 构建消息列表
    messages = []
    for msg in context:
        messages.append(TextMessage(content=msg["content"], source=msg["source"]))
    
    # 调用专家获取回复
    response = await expert.on_messages(messages, cancellation_token=None)
    return response.chat_message.content


async def human_intervention_step(current_speaker: str, round_num: int, total_rounds: int):
    """
    人工干预步骤 - 在每位专家发言后暂停等待人工输入
    
    Returns:
        (should_continue, user_input): 是否继续，用户输入内容
    """
    print(f"\n{'─' * 60}")
    print(f"🎤 {current_speaker} 刚刚发言完毕")
    print(f"📊 当前进度：第 {round_num}/{total_rounds} 轮")
    print(f"{'─' * 60}")
    print("💡 请选择操作：")
    print("   1. 输入你的想法 → 插话补充")
    print("   2. 直接回车 → 跳过，继续下一位专家")
    print("   3. 输入「结束」→ 终止整个会议")
    print(f"{'─' * 60}")
    
    try:
        user_input = input("你的选择：").strip()
    except EOFError:
        user_input = ""
    
    if user_input.lower() in ["结束", "end", "stop", "quit", "exit"]:
        return False, None
    
    return True, user_input if user_input else None


async def run_brainstorm(topic: str, max_rounds: int = 10):
    """
    运行头脑风暴会议 - 逐个调用专家，确保人工介入
    
    Args:
        topic: 讨论主题
        max_rounds: 最大讨论轮数
    """
    print("=" * 60)
    print("🚀 多专家头脑风暴会议已启动")
    print("=" * 60)
    print(f"\n📋 讨论主题：{topic}")
    print(f"🎯 最大轮数：{max_rounds} 轮")
    print("\n💡 系统配置：")
    print(f"   • API: {os.getenv('OPENAI_API_BASE', 'https://www.dmxapi.cn/v1')}")
    print("=" * 60)
    
    # 1. 创建专家Agent
    print("\n🤖 正在初始化专家...")
    expert_agents = create_expert_agents()
    expert_names = [agent.name for agent in expert_agents]
    print(f"\n👥 参会专家：{', '.join(expert_names)}")
    
    # 2. 初始化对话历史
    all_messages = []
    
    # 3. 启动消息
    initial_message = f"""各位专家，让我们围绕「{topic}」进行头脑风暴。

讨论规则：
1. 自由发言、互相补充、不批评、尽量发散
2. 每位专家仅从自身领域发言，不跨领域
3. 发言简洁有力，2-3句话即可

主持人（你）会在每位专家发言后介入，可以选择插话、跳过或结束讨论。

请开始第一轮发言！"""

    print("\n🎯 主持人（你）已发起讨论\n")
    print(f"📝 初始话题：{topic}\n")
    
    # 添加系统消息到历史
    all_messages.append({"source": "Host", "content": initial_message})
    
    # 4. 手动控制循环 - 逐个专家调用
    round_num = 1
    
    while round_num <= max_rounds:
        print(f"\n{'=' * 60}")
        print(f"🔄 第 {round_num} 轮讨论开始")
        print(f"{'=' * 60}")
        
        # 每个专家发言一次
        for expert in expert_agents:
            print(f"\n⏳ {expert.name} 正在思考...")
            
            try:
                # 调用专家获取回复
                response = await get_expert_response(expert, all_messages)
                
                print(f"\n💬 [{expert.name}]: {response}\n")
                all_messages.append({"source": expert.name, "content": response})
                
                # 专家发言后，人工介入
                should_continue, user_input = await human_intervention_step(
                    expert.name, round_num, max_rounds
                )
                
                if not should_continue:
                    print("\n🛑 会议被主持人终止")
                    save_chat_history(all_messages)
                    return all_messages
                
                # 如果用户有输入，添加到历史
                if user_input:
                    all_messages.append({"source": "Host", "content": user_input})
                    print(f"\n🎤 [主持人]: {user_input}")
                    
            except Exception as e:
                print(f"\n❌ {expert.name} 发言出错: {e}")
                continue
        
        round_num += 1
        
        # 检查是否继续下一轮
        if round_num <= max_rounds:
            print(f"\n{'─' * 60}")
            cont = input(f"第 {round_num-1} 轮结束，是否继续第 {round_num} 轮？(回车继续/输入'结束'停止): ").strip()
            if cont.lower() in ["结束", "end", "stop", "n", "no"]:
                print("\n🛑 会议被主持人终止")
                break
    
    # 5. 保存对话历史
    save_chat_history(all_messages)
    
    print("\n" + "=" * 60)
    print("🎉 头脑风暴会议已结束")
    print("=" * 60)
    
    return all_messages


def main():
    """主入口"""
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误：未设置 OPENAI_API_KEY 环境变量")
        print("请检查 .env 文件或设置环境变量")
        return
    
    print("\n🧠 AutoGen 多专家头脑风暴系统")
    print("-" * 40)
    
    # 可以使用默认主题或自定义
    default_topic = "AI驱动的个人知识管理工具"
    user_input = input(f"\n请输入讨论主题（直接回车使用默认主题：{default_topic}）：").strip()
    topic = user_input if user_input else default_topic
    
    # 获取轮数
    rounds_input = input("请输入最大讨论轮数（直接回车默认3轮）：").strip()
    max_rounds = int(rounds_input) if rounds_input.isdigit() else 3
    
    # 运行头脑风暴
    asyncio.run(run_brainstorm(topic, max_rounds))


if __name__ == "__main__":
    main()
