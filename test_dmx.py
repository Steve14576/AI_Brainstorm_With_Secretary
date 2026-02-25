"""DMX API 测试脚本 - 调通 OpenAI 兼容接口。

目的：
- 验证 DMX API 的返回结构
- 找出为什么会报 'str' object has no attribute 'choices' 
- 确定正确的调用方式

运行：uv run python test_dmx.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("MODEL_NAME", "glm-5")

print(f"[配置] API Key: {api_key[:20] if api_key else 'None'}...")
print(f"[配置] Base URL: {base_url}")
print(f"[配置] Model: {model_name}")

if not api_key or not base_url:
    print("\n[错误] 请在 .env 中配置 OPENAI_API_KEY 和 OPENAI_API_BASE")
    exit(1)

print("\n=== 测试 1：基本调用 ===")
try:
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": "请用一句话介绍你自己。"}
        ],
        temperature=0.7,
    )
    
    print(f"[成功] 响应类型: {type(response)}")
    print(f"[成功] 响应对象: {response}")
    print(f"\n[内容] {response.choices[0].message.content}")
    
except Exception as e:
    print(f"[失败] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试 2：检查响应结构 ===")
try:
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "说一个字：好"}],
        temperature=0.1,
    )
    
    print(f"响应类型: {type(response)}")
    print(f"是否有 choices 属性: {hasattr(response, 'choices')}")
    
    if hasattr(response, 'choices'):
        print(f"choices 类型: {type(response.choices)}")
        print(f"choices 长度: {len(response.choices)}")
        print(f"第一个 choice: {response.choices[0]}")
        print(f"message: {response.choices[0].message}")
        print(f"content: {response.choices[0].message.content}")
    else:
        print(f"响应内容: {response}")
        print(f"响应的所有属性: {dir(response)}")
        
except Exception as e:
    print(f"[失败] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试 3：用于会议系统的实际调用 ===")
try:
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    system_prompt = "你是一名产品负责人，擅长从用户和业务角度思考。"
    user_message = "当前议程是：产品方向与目标用户。请简要给出一段立场声明（不超过150字），说明你认为本议程最关键要解决的问题是什么。"
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        top_p=0.9,
    )
    
    content = response.choices[0].message.content
    print(f"[专家立场声明]:\n{content}")
    
except Exception as e:
    print(f"[失败] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
