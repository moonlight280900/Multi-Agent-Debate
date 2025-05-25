"""
意图识别Agent
"""

from typing import Dict, Any, List, Optional
import json

from .base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI


class IntentAgent(BaseAgent):
    """
    意图识别Agent，负责识别视频中的人类动作意图
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化意图识别Agent
        
        Args:
            name: Agent名称
            config: 配置参数
        """
        super().__init__(name, config)
        
        # 初始化LLM - 使用自定义API端点
        api_key = config.get("openai_api_key", "sk-gOP770ea8c4de8e72313b5efd794303cad1a0c1cf06Eumg1")
        base_url = config.get("openai_base_url", "https://api.gptsapi.net/v1")
        
        # 使用LangChain的ChatOpenAI，并配置自定义API端点
        self.model = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.1
        )
        
        # 直接创建OpenAI客户端作为备用
        self.openai_client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # 支持的动作意图类别
        self.action_intents = config.get("action_intents", [])
        self.logger.info(f"IntentAgent已初始化，支持{len(self.action_intents)}种动作意图类别")
        
        # 创建意图识别提示模板
        self.intent_prompt = ChatPromptTemplate.from_template("""
You are an advanced human behavior analysis and intent understanding system. Your task is to analyze video content and understand the essence and purpose behind human behaviors, providing deep insights.

Below is the description of video content:
{frame_analyses}

Please provide a comprehensive behavioral analysis that goes beyond simple classification. Your analysis should delve into the nature and purpose of the behaviors observed. Include the following aspects:

1. Current Behavior:
   - Detailed description of the specific action sequences observed
   - Precision, force, speed, and fluidity of movements
   - Professional quality and skill level demonstrated

2. Behavioral Intent & Motivation:
   - The immediate purpose behind these actions
   - Potential long-term goals and intentions
   - Possible internal motivations (physiological needs, social needs, self-actualization, etc.)

3. Contextual & Environmental Factors:
   - How the environment influences or facilitates this behavior
   - Impact of social settings and situations on the behavior
   - Cultural factors shaping behavioral patterns

4. Cognitive & Decision Processes:
   - Likely thought processes during the behavior
   - Balance between planned and spontaneous elements in the behavior
   - Key decision points and behavior adjustments

5. Predicted Next Actions:
   - 3-5 potential subsequent actions based on behavioral patterns
   - Assessment of the likelihood and rationale for each prediction
   - Key factors influencing the choice of subsequent behaviors

6. Goals & Outcomes Analysis:
   - Ultimate goals the behavior may achieve
   - Expected and actual outcomes of the behavior
   - Evaluation of efficiency and probability of success

7. Behavioral Patterns & Personal Characteristics:
   - Habitual patterns and personality traits reflected in the behavior
   - Professional background or training evident in behavioral style
   - Emotional states and psychological traits revealed through behavior

Please respond in the following JSON format, ensuring your analysis is thorough, insightful, and perceptive:

```json
{{
    "current_action": "Detailed and precise description of the observed action sequence, including specifics and techniques",
    "behavioral_intent": "In-depth analysis of immediate purpose and long-term intentions behind the behavior",
    "contextual_factors": "Analysis of environmental and social factors influencing and constraining the behavior",
    "cognitive_process": "Inference about the subject's thinking patterns and decision-making during the behavior",
    "predicted_next_actions": [
        "First potential subsequent action based on behavioral patterns",
        "Second potential subsequent action",
        "Third potential subsequent action",
        "Fourth potential subsequent action",
        "Fifth potential subsequent action"
    ],
    "goal_analysis": "Analysis of the ultimate goals the behavior is directed toward and possible outcomes",
    "confidence_level": 0.95,
    "behavioral_pattern": "Interpretation of habitual patterns, training background, and personality traits shown in the behavior",
    "emotional_state": "Emotional state and psychological tendencies inferred from the behavior",
    "social_significance": "Meaning and value of the behavior in social interactions"
}}
```

Ensure your analysis is based on actual evidence from the video description rather than subjective assumptions. Strive to provide unique and profound insights that reveal behavioral details and psychological motivations that might be overlooked by casual observers.
""")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理视频分析结果，识别动作意图
        
        Args:
            input_data: 视频分析结果
            
        Returns:
            意图识别结果
        """
        # 检查分析类型并提取相应的分析结果
        analysis_type = input_data.get("analysis_type", "frame_by_frame")
        
        frame_analyses = []
        if analysis_type == "direct_video":
            # 直接视频分析模式
            content = input_data.get("content", "")
            if content:
                frame_analyses.append(f"视频分析:\n{content}\n")
                self.logger.info(f"处理直接视频分析结果")
            else:
                return {
                    "success": False,
                    "error": "视频分析结果为空"
                }
        else:
            # 逐帧分析模式
            frame_results = input_data.get("frame_results", [])
            
            if not frame_results:
                return {
                    "success": False,
                    "error": "没有找到有效的视频帧分析结果"
                }
            
            self.logger.info(f"开始处理{len(frame_results)}个视频帧的分析结果")
            
            # 提取所有成功的帧分析文本
            for i, frame_result in enumerate(frame_results):
                result = frame_result.get("result", {})
                if result.get("success", False):
                    content = result.get("content", "")
                    frame_analyses.append(f"帧 {i+1}:\n{content}\n")
        
        if not frame_analyses:
            return {
                "success": False,
                "error": "所有视频分析都失败了"
            }
        
        try:
            # 格式化动作意图类别
            action_intents_str = "\n".join([f"- {intent}" for intent in self.action_intents])
            
            # 使用LLM识别意图
            input_values = {
                "action_intents": action_intents_str,
                "frame_analyses": "\n".join(frame_analyses)
            }
            
            try:
                # 首先尝试使用LangChain
                chain = self.intent_prompt | self.model
                llm_response = await chain.ainvoke(input_values)
                llm_response_text = llm_response.content
            except Exception as chain_error:
                self.logger.warning(f"使用LangChain调用失败: {str(chain_error)}，尝试直接使用OpenAI客户端")
                
                # 如果LangChain失败，直接使用OpenAI客户端
                prompt_text = self.intent_prompt.format(**input_values)
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "你是一个专业的人类动作意图识别系统。"},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=0.1
                )
                llm_response_text = response.choices[0].message.content
            
            # 提取JSON
            try:
                # 如果响应被包裹在```json...```中
                if "```json" in llm_response_text and "```" in llm_response_text.split("```json", 1)[1]:
                    json_str = llm_response_text.split("```json", 1)[1].split("```", 1)[0]
                    intent_result = json.loads(json_str)
                else:
                    # 尝试直接解析
                    intent_result = json.loads(llm_response_text)
                
                # 更新状态
                self.update_state({
                    "analysis_type": analysis_type,
                    "processed_content": len(frame_analyses),
                    "intent_result": intent_result
                })
                
                return {
                    "success": True,
                    "intent_result": intent_result,
                    "message": f"成功进行行为分析"
                }
                
            except json.JSONDecodeError:
                # 如果无法解析JSON，返回原始文本
                return {
                    "success": True,
                    "raw_response": llm_response_text,
                    "message": "成功获取响应，但无法解析为JSON格式"
                }
                
        except Exception as e:
            self.logger.error(f"识别动作意图时出错: {str(e)}")
            return {
                "success": False,
                "error": f"识别动作意图时出错: {str(e)}"
            } 