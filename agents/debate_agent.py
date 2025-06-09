import os
import asyncio
from typing import Dict, Any, List, Optional
import json
import torch  # Added for GPU memory management

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
        self.perspective = perspective
        model_path = config.get("model_path", "TinyLLaVA-Video-R1")
        self.model = config.get("shared_model") or self.load_model(model_path)
        self.max_frames = config.get("max_frames", 16)
        self.logger.info(f"辩论Agent '{name}' 已初始化，视角: {perspective}")
    
    def load_model(self, model_path: str) -> TinyLLaVA:
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        try:
            self.logger.info(f"正在加载模型: {model_path}")
            model = TinyLLaVA(model_path=model_path)
            self.logger.info("模型加载成功")
            return model
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
            return {"success": False, "error": "模型未加载"}
        prompt = self._get_perspective_prompt()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.model.analyze_video(video_path, prompt, num_frames=self.max_frames))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory
        return result
    
    def _get_perspective_prompt(self) -> str:
        """
        根据Agent的视角/立场生成相应的提示词
        
        Returns:
            针对特定视角的提示词
        """
        base_prompt = """
        You are an expert in human action analysis. Carefully examine the following video, focusing on physical movements, environmental interactions, and behavioral sequences. Reflect on their relationship to goals and intentions. Enclose your reasoning process within <think> </think> tags, followed by your final conclusion.
        """

        perspective_prompts = {
            "action_focused": """
            - Observe and describe physical movements.
            - Detail the sequence, timing, and precision of actions.
            - Analyze interaction with objects and the environment.
            - Evaluate the technical skill demonstrated.

            Focus strictly on observable behaviors. Analyze:
            1. Distinct motor actions.
            2. Temporal order and rhythm.
            3. Use and manipulation of objects or space.
            4. Repetitive or patterned behaviors.
            5. Proficiency and control.

            Avoid interpreting internal states or intentions.
            """,
            "intention_focused": """
            - Hypothesize possible goals and motivations.
            - Interpret actions in light of inferred intentions.
            - Use non-verbal cues to assess purpose.
            - Distinguish between planned vs. spontaneous behaviors.

            Focus on intentionality. Analyze:
            1. Underlying goals driving the actions.
            2. Contextual motivation (short and long term).
            3. Evidence of planning or improvisation.
            4. Gestures and cues revealing intent.
            5. Multiple plausible interpretations.

            Support all inferences with observed behavior.
            """,
            "context_focused": """
            - Assess the physical and social setting.
            - Explore how environment influences behavior.
            - Consider cultural and situational norms.
            - Identify key contextual elements affecting the action.

            Focus on contextual influence. Analyze:
            1. Relationship between setting and behavior.
            2. Presence and role of objects/tools.
            3. Impact of social surroundings.
            4. Cultural conventions shaping actions.
            5. How behavior might change in other contexts.
            """,
            "emotional_focused": """
            - Detect emotional states through body language and facial expression.
            - Trace emotional dynamics throughout the action.
            - Assess how emotions influence decisions and behavior.

            Focus on affective aspects. Analyze:
            1. Observable emotional cues.
            2. Temporal shifts in emotion.
            3. Connection between emotions and actions.
            4. Regulation or suppression of emotional responses.
            5. Emotional context of decision-making.

            Cite specific behavioral evidence where possible.
            """,
            "social_focused": """
            - Examine interactions among individuals.
            - Consider social norms, roles, and expectations.
            - Evaluate the social function of behaviors.

            Focus on the social dimension. Analyze:
            1. Communication signals and their meanings.
            2. Role-based behaviors and group dynamics.
            3. Conformity to or defiance of norms.
            4. Interpersonal influence and reactions.
            5. Broader social interpretation of actions.

            """
        }
        return base_prompt + (perspective_prompts.get(self.perspective) or """
        - Analyze actions, intentions, context, emotions, social factors.
        - Identify key movements.
        - Infer intentions.
        - Consider environmental/social influence.
        - Observe emotions.
        Provide a holistic analysis integrating multiple perspectives.
        """)
    
    async def respond_to_argument(self, argument: str) -> Dict[str, Any]:
        """
        响应其他Agent的辩论观点
        
        Args:
            argument: 其他Agent提出的辩论观点
            
        Returns:
            响应结果
        """
        prompt = f"""
        You are an expert analyzing human behavior from a {self.perspective} perspective. Review the following interpretation from another analyst:

        "{argument}"

        Your task is to critically engage with the argument by:
        - Relating it to your analytical lens.
        - Identifying areas of agreement and divergence.
        - Referencing relevant behavioral or contextual evidence.
        - Deepening or refining the analysis.
        - If applicable, connecting to prior debate contributions.

        Structure your response by:
        1. Recognizing valid observations.
        2. Highlighting gaps or limitations.
        3. Adding nuanced insights.
        4. Proposing an integrative understanding if possible.

        Maintain a constructive and professional tone aimed at collaborative understanding.
        """

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.model.analyze_text(prompt))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory
        if result.get("success", False):
            return {"success": True, "perspective": self.perspective, "response": result.get("response", ""), "agent_name": self.name}
        else:
            self.logger.error(f"处理辩论回应失败: {result.get('error', '未知错误')}")
            return {"success": False, "error": result.get("error", "处理辩论回应失败")}
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入数据
        
        Args:
            input_data: 输入数据，可以是视频路径或其他Agent的辩论观点
            
        Returns:
            处理结果
        """
        if "video_path" in input_data:
            result = await self.process_video(input_data["video_path"])
            print(f"agent is {self.perspective}, and its result is {result}")
            if result.get("success", False):
                result["agent_perspective"] = self.perspective
                result["agent_name"] = self.name
            return result
        elif "argument" in input_data:
            return await self.respond_to_argument(input_data["argument"])
        else:
            return {"success": False, "error": "输入数据必须包含video_path或argument"}