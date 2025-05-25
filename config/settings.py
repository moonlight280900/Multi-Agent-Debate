"""
系统配置设置
"""

import os
from pydantic import BaseModel
from typing import List

class Settings(BaseModel):
    """系统配置类"""
    
    # API密钥和端点
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "sk-gOP770ea8c4de8e72313b5efd794303cad1a0c1cf06Eumg1")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.gptsapi.net/v1")
    
    # TinyLLaVA配置
    model_name: str = "openai/tinyLLaVA-video-r1"
    model_path: str = os.getenv("TINYLLAVA_MODEL_PATH", "TinyLLaVA-Video-R1")
    
    # Agent配置
    agent_timeout: int = 60  # 秒
    max_retries: int = 3
    
    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 意图类别
    action_intents: List[str] = [
        "走路/行走", "跑步", "跳跃", "蹲下", "站立",
        "坐下", "挥手", "拍手", "拥抱", "推动",
        "拉动", "抬起物体", "放下物体", "扔东西", "抓取",
        "敲击", "指向", "打电话", "写字", "阅读",
        "吃饭", "喝水", "穿衣", "脱衣", "打开门/窗",
        "关闭门/窗", "驾驶", "骑行", "滑行", "游泳"
    ] 
