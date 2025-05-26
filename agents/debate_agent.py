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
        As an expert in human action analysis, examine this video carefully. Pay attention to physical movements, environmental interactions, and behavior sequences. Consider their relation to intentions and goals. Output your thinking in <think> </think> tags.
        """
        perspective_prompts = {
            "action_focused": """
            <think>
            - Identify physical movements.
            - Describe sequence, speed, precision.
            - Analyze interaction with environment/objects.
            - Assess skill level.
            </think>
            Focus on objective action description. Analyze:
            1. Specific movements.
            2. Sequence and timing.
            3. Environmental interaction.
            4. Patterns or repetitions.
            5. Technical skills demonstrated.
            Avoid speculation on intentions or emotions.
            """,
            "intention_focused": """
            <think>
            - Infer possible intentions.
            - Consider goals and motivations.
            - Analyze non-verbal cues for intent.
            - Explore short-term and long-term objectives.
            </think>
            Focus on intentions. Analyze:
            1. Likely goals.
            2. Motivations.
            3. Planned or spontaneous behavior.
            4. Non-verbal cues suggesting intent.
            5. Alternative explanations.
            Support with video evidence.
            """,
            "context_focused": """
            <think>
            - Analyze physical setting.
            - Consider environmental influence.
            - Think about social/cultural context.
            - Observe interactions with others.
            </think>
            Focus on context's role. Analyze:
            1. Physical setting's relation to actions.
            2. Central objects/tools.
            3. Social environment's effect.
            4. Cultural/situational norms.
            5. Behavior differences in other contexts.
            """,
            "emotional_focused": """
            <think>
            - Observe facial expressions and body language.
            - Analyze emotional cues.
            - Note emotional changes.
            - Consider emotion's influence on actions.
            </think>
            Focus on emotions. Analyze:
            1. Evident emotions.
            2. Relation to actions.
            3. Emotional shifts and triggers.
            4. Impact on behavior/decision-making.
            5. Evidence of emotion regulation.
            Link emotions to actions where possible.
            """,
            "social_focused": """
            <think>
            - Identify social interactions.
            - Consider social norms/expectations.
            - Think about social roles/identities.
            - Analyze behavior's social perception.
            </think>
            Focus on social dimensions. Analyze:
            1. Behavior's social function.
            2. Social messages/signals.
            3. Conformity to/challenge of norms.
            4. Enacted social roles.
            5. Potential reactions/interpretations.
            Provide social significance analysis.
            """
        }
        return base_prompt + (perspective_prompts.get(self.perspective) or """
        <think>
        - Analyze actions, intentions, context, emotions, social factors.
        - Identify key movements.
        - Infer intentions.
        - Consider environmental/social influence.
        - Observe emotions.
        </think>
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
        As an expert from the {self.perspective} perspective, review this interpretation:
        
        "{argument}"
        
        <think>
        - Relate to your perspective.
        - Identify agreement/disagreement.
        - Consider video evidence.
        - Add depth/nuance.
        - If part of ongoing debate, build on previous discussions.
        </think>
        
        Respond by:
        1. Acknowledging valid points.
        2. Noting limitations/oversights.
        3. Offering additional insights.
        4. Suggesting integrated understanding.
        
        Maintain a collaborative tone.
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