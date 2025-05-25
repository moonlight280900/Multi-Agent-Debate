"""
模型Agent - 专门负责处理TinyLLaVA模型的调用和输出
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
import json

from .base_agent import BaseAgent
from models.tinyllava import TinyLLaVA


class ModelAgent(BaseAgent):
    """
    模型Agent，专门负责处理TinyLLaVA-Video-R1模型的调用和输出处理
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化模型Agent
        
        Args:
            name: Agent名称
            config: 配置参数
        """
        super().__init__(name, config)
        
        # 模型路径
        model_path = config.get("model_path", "TinyLLaVA-Video-R1")
        
        # 初始化模型 - 检查是否有共享模型实例
        self.model = None
        if "shared_model" in config and config["shared_model"] is not None:
            self.logger.info("使用共享模型实例")
            self.model = config["shared_model"]
        else:
            # 如果没有共享模型，则加载新的模型实例
            self.load_model(model_path)
        
        # 模型参数设置
        self.temperature = config.get("temperature", 0.1)
        self.top_p = config.get("top_p", None)
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.num_beams = config.get("num_beams", 1)
        
        # 视频处理参数
        self.max_frames = config.get("max_frames", 16)
        
        self.logger.info(f"ModelAgent已初始化，使用模型: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        try:
            self.logger.info(f"正在加载模型: {model_path}")
            self.model = TinyLLaVA(model_path=model_path)
            self.logger.info("模型加载成功")
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    async def process_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            prompt: 提示词
            
        Returns:
            处理结果
        """
        self.logger.info(f"处理图像: {image_path}")
        
        if not self.model:
            return {
                "success": False,
                "error": "模型未加载"
            }
        
        # 调用模型分析图像
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.model.analyze_video_frame(image_path, prompt)
        )
        
        return result
    
    async def process_video(self, video_path: str, prompt: str) -> Dict[str, Any]:
        """
        处理视频
        
        Args:
            video_path: 视频路径
            prompt: 提示词
            
        Returns:
            处理结果
        """
        self.logger.info(f"处理视频: {video_path}")
        
        if not self.model:
            return {
                "success": False,
                "error": "模型未加载"
            }
        
        # 调用模型分析视频
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.model.analyze_video(video_path, prompt, num_frames=self.max_frames)
        )
        
        return result
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入数据
        
        Args:
            input_data: 输入数据
            
        Returns:
            处理结果
        """
        # 检测输入类型并分发到对应处理方法
        if "image_path" in input_data:
            # 图像处理
            prompt = input_data.get("prompt", """Please provide a detailed description of this image, including:
1. Scene context: The environment, location, and time of day
2. Main subjects: Detailed appearance of the primary people or objects, including clothing and distinctive features
3. Action analysis: Specific actions being performed, their manner and characteristics
4. Details of importance: Background elements, environmental factors, and small but significant details
5. Emotional expressions: Facial expressions and body language that reveal emotional states
6. Interaction patterns: If multiple subjects are present, how they interact with each other
7. Technical elements: Camera angle, lighting conditions, and other factors that contribute to understanding the image

Ensure your description is comprehensive, accurate, and captures the essential information and essence of the image without missing important details.""")
            result = await self.process_image(input_data["image_path"], prompt)
        elif "video_path" in input_data:
            # 视频处理
            prompt = input_data.get("prompt", """Please provide a comprehensive analysis of this video, including:
1. Scene setting: The environment, location, and any changes throughout the video
2. Subject description: The appearance, clothing, features, and identity characteristics of the main people
3. Core actions: Detailed description of the primary action sequence, including the initiation, process, and completion
4. Action intent: Analysis of the motivation and purpose behind the observed behaviors
5. Technical execution: Assessment of skill level, proficiency, and special techniques displayed
6. Interaction patterns: How subjects interact with the environment, objects, or other people
7. Temporal progression: The time sequence and development of events
8. Object details: The appearance, function, and role of important objects in the video
9. Emotional expression: Emotional states conveyed through facial expressions, posture, and actions
10. Contextual background: Possible social or cultural context, type of activity, and purpose

Provide thorough analysis, ensuring you capture key information, behavioral patterns, and the essence of the content, with particular attention to human intent and purpose.Output the thinking process in <think> </think>""")
            result = await self.process_video(input_data["video_path"], prompt)
        else:
            return {
                "success": False,
                "error": "输入数据必须包含image_path或video_path"
            }
        
        # 更新Agent状态
        if result.get("success", False):
            self.update_state({
                "last_input": input_data,
                "last_result": result
            })
        
        return result
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        根据提示词生成回答
        
        Args:
            prompt: 提示词
            context: 上下文信息
            
        Returns:
            生成结果
        """
        if not context:
            context = {}
        
        # 这里可以实现一个简单的文本生成功能，但由于TinyLLaVA主要处理多模态输入，
        # 所以这个功能可能不是核心需求，我们这里仅返回一个简单的响应
        return {
            "success": True,
            "content": f"ModelAgent准备就绪，可以处理图像和视频输入。请提供多模态内容。",
            "context": context
        } 