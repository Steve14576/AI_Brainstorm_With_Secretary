"""
环境测试脚本
验证 AutoGen 是否可以正常工作
"""

import os
from dotenv import load_dotenv

def test_environment():
    """测试环境配置"""
    print("="*60)
    print("环境检查")
    print("="*60)
    
    # 加载环境变量
    load_dotenv()
    
    # 检查必需的环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    
    print(f"✓ API Base: {api_base}")
    print(f"✓ Model Name: {model_name}")
    
    if not api_key or api_key == "your_api_key_here":
        print("\n⚠️  警告: 尚未配置 OPENAI_API_KEY")
        print("   请复制 .env.example 为 .env 并填入你的 API key")
        return False
    else:
        print(f"✓ API Key: {api_key[:8]}...{api_key[-4:]}")
    
    # 尝试导入 AutoGen
    try:
        import autogen
        print(f"\n✓ AutoGen 版本: {autogen.__version__}")
        print("✓ AutoGen 导入成功")
    except Exception as e:
        print(f"\n✗ AutoGen 导入失败: {e}")
        return False
    
    print("\n" + "="*60)
    print("环境配置完成！可以开始使用了。")
    print("="*60)
    return True


if __name__ == "__main__":
    test_environment()
