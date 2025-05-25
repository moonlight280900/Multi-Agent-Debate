"""
辩论Agent - 负责从不同角度分析视频中的人类行为意图
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
import json

from .base_agent import BaseAgent
from models.tinyllava import TinyLLaVA


class DebateAgent(BaseAgent):
    """
    辩论Agent，基于TinyLLaVA-Video-R1模型，从特定角度分析视频中的人类行为意图
    """
    
    def __init__(self, name: str, perspective: str, config: Dict[str, Any]):
        """
        初始化辩论Agent
        
        Args:
            name: Agent名称
            perspective: Agent的视角/辩论立场
            config: 配置参数
        """
        super().__init__(name, config)
        
        # 设置Agent的视角/辩论立场
        self.perspective = perspective
        
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
        
        # 视频处理参数
        self.max_frames = config.get("max_frames", 16)
        
        self.logger.info(f"辩论Agent '{name}' 已初始化，视角: {perspective}")
    
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
    
    async def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        从特定视角处理视频
        
        Args:
            video_path: 视频路径
            
        Returns:
            处理结果
        """
        self.logger.info(f"从视角 '{self.perspective}' 处理视频: {video_path}")
        
        if not self.model:
            return {
                "success": False,
                "error": "模型未加载"
            }
        
        # 根据不同视角/立场构建不同的提示词
        prompt = self._get_perspective_prompt()
        
        # 调用模型分析视频
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.model.analyze_video(video_path, prompt, num_frames=self.max_frames)
        )
        
        return result
    
    def _get_perspective_prompt(self) -> str:
        """
        根据Agent的视角/立场生成相应的提示词
        
        Returns:
            针对特定视角的提示词
        """
        # 基础提示词部分
        base_prompt = """
        As an expert in human action analysis, examine this video carefully.
        Output the thinking process in <think> </think> tags.
        """
        
        # 根据不同视角/立场添加特定的提示内容
        if self.perspective == "action_focused":
            perspective_prompt = """
            <think>
            - First, I should identify all visible physical movements in the video
            - I need to note the sequence, speed, and precision of actions
            - I should pay attention to what body parts are involved in the actions
            - Are these actions skilled/trained or spontaneous?
            - How do these actions relate to objects in the environment?
            - What physical goals might these actions be trying to achieve?
            </think>
            
            Focus PRIMARILY on the PHYSICAL ACTIONS being performed. Analyze:
            1. What specific physical movements and actions are occurring?
            2. What is the sequence, timing, and coordination of these movements?
            3. How skillful, deliberate, or spontaneous are these actions?
            4. What objects are being physically manipulated?
            5. What technical aspects of the movements suggest training or experience?
            
            Provide a detailed physical action analysis while avoiding psychological interpretation or intent speculation.
            """
        
        elif self.perspective == "intention_focused":
            perspective_prompt = """
            <think>
            - I need to look beyond the physical actions to identify possible intentions
            - What goals might the person be trying to achieve?
            - Are there clues in facial expressions that suggest intentions?
            - How does the environment context help understand the person's purpose?
            - Are there multiple possible interpretations of the intentions?
            - What long-term vs short-term goals might be present?
            </think>
            
            Focus PRIMARILY on the INTENTIONS behind the actions. Analyze:
            1. What goals or objectives is the person likely trying to accomplish?
            2. What motivations might explain these actions?
            3. Is the person acting with purpose or spontaneously?
            4. What does facial expression and body language reveal about intent?
            5. Are there multiple possible intentions that could explain the behavior?
            
            Provide a detailed intention analysis while considering multiple possible interpretations.
            """
            
        elif self.perspective == "context_focused":
            perspective_prompt = """
            <think>
            - I should analyze the environment and setting in detail
            - How does the location influence or constrain the actions?
            - What social or cultural context might be relevant?
            - Are there other people present and how does this affect behavior?
            - What objects in the environment are significant to understanding the actions?
            - How does time of day or other environmental factors matter?
            </think>
            
            Focus PRIMARILY on the CONTEXT and ENVIRONMENT of the actions. Analyze:
            1. What is the physical setting and how does it relate to the behavior?
            2. What social or cultural context might be influencing the actions?
            3. How do environmental factors constrain or enable the behavior?
            4. What objects in the environment are significant to understanding the situation?
            5. How might this behavior differ in a different context?
            
            Provide a detailed contextual analysis that explains how environment shapes the observed behavior.
            """
            
        elif self.perspective == "emotional_focused":
            perspective_prompt = """
            <think>
            - I need to carefully observe facial expressions and micro-expressions
            - Body language can reveal emotional states - posture, tension, etc.
            - Voice tone and speech patterns (if audible) provide emotional clues
            - Are there emotional changes throughout the video?
            - How might emotions be influencing the actions?
            - Are emotions being regulated or expressed freely?
            </think>
            
            Focus PRIMARILY on the EMOTIONAL ASPECTS of the behavior. Analyze:
            1. What emotions are displayed through facial expressions and body language?
            2. How do emotions appear to influence the actions being performed?
            3. Are there changes in emotional state throughout the video?
            4. How might the person's emotional state affect their decision-making?
            5. Is there evidence of emotion regulation or emotional reactions?
            
            Provide a detailed emotional analysis that explains how feelings might be driving behavior.
            """
            
        elif self.perspective == "social_focused":
            perspective_prompt = """
            <think>
            - Are there interactions with other people in the video?
            - How do social dynamics influence the behavior?
            - Is this behavior meant to communicate something to others?
            - Does the behavior follow or violate social norms?
            - What social roles might the person be fulfilling?
            - How might observers perceive or interpret this behavior?
            </think>
            
            Focus PRIMARILY on the SOCIAL DIMENSIONS of the behavior. Analyze:
            1. How does this behavior function within social interactions or relationships?
            2. What social messages might this behavior be communicating?
            3. Does the behavior conform to or violate social norms?
            4. What social roles or identities is the person expressing?
            5. How might this behavior influence others' perceptions or responses?
            
            Provide a detailed social analysis that explains the interpersonal significance of the behavior.
            """
        
        else:
            # 默认通用视角
            perspective_prompt = """
            <think>
            - I should analyze both the physical actions and potential intentions
            - The context and environment provide important clues
            - Emotional expressions and social factors should be considered
            - I should consider multiple possible interpretations of the behavior
            </think>
            
            Provide a comprehensive analysis of this video, including:
            1. The specific physical actions being performed
            2. Likely intentions and motivations behind these actions
            3. How the environment and context influence the behavior
            4. Relevant emotional states and social dimensions
            5. Multiple possible interpretations of the observed behavior
            
            Analyze the video from multiple perspectives to provide a balanced view.
            """
        
        return base_prompt + perspective_prompt
    
    async def respond_to_argument(self, argument: str) -> Dict[str, Any]:
        """
        响应其他Agent的辩论观点
        
        Args:
            argument: 其他Agent提出的辩论观点
            
        Returns:
            响应结果
        """
        # 构建提示词，要求模型从当前Agent的视角回应其他Agent的观点
        prompt = f"""
        As an expert analyzing human behavior from the {self.perspective} perspective, 
        review and respond to this alternative interpretation:
        
        "{argument}"
        
        Output the thinking process in <think> </think> tags.
        
        - you need to consider how this interpretation aligns or conflicts with your perspective
        - What aspects might this interpretation be missing from your viewpoint?
        - Are there areas where you agree with this interpretation?
        - What evidence from the video supports or challenges this view?
        - How can you constructively add to or refine this interpretation?
        
        Provide a response that:
        1. Acknowledges valid points in the other interpretation
        2. Identifies aspects that might be overlooked from your perspective
        3. Offers additional insights based on your specialized focus
        4. Suggests how combining perspectives might lead to a more complete understanding
        
        Maintain a collaborative tone while advocating for your perspective's importance.
        """
        
        # 使用模型的analyze_text方法处理纯文本输入
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.model.analyze_text(prompt)
        )
        
        if result.get("success", False):
            return {
                "success": True,
                "perspective": self.perspective,
                "response": result.get("response", ""),
                "agent_name": self.name
            }
        else:
            self.logger.error(f"处理辩论回应失败: {result.get('error', '未知错误')}")
            return {
                "success": False,
                "error": result.get("error", "处理辩论回应失败")
            }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入数据
        
        Args:
            input_data: 输入数据，可以是视频路径或其他Agent的辩论观点
            
        Returns:
            处理结果
        """
        # 判断输入类型
        if "video_path" in input_data:
            # 处理视频
            result = await self.process_video(input_data["video_path"])
            print(f"""agent is {self.perspective},and its result is {result}""")
            # 添加Agent视角信息
            if result.get("success", False):
                result["agent_perspective"] = self.perspective
                result["agent_name"] = self.name
            
            return result
            
        elif "argument" in input_data:
            # 响应其他Agent的辩论观点
            return await self.respond_to_argument(input_data["argument"])
            
        else:
            return {
                "success": False,
                "error": "输入数据必须包含video_path或argument"
            } 