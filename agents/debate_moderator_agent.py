"""
辩论主持Agent - 负责协调多个辩论Agent讨论并总结最终的意图识别结果
"""

import asyncio
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime

from .base_agent import BaseAgent
from models.tinyllava import TinyLLaVA


class DebateModeratorAgent(BaseAgent):
    """
    辩论主持Agent，负责协调多个辩论Agent之间的讨论，并总结最终的人类动作意图识别结果
    使用本地TinyLLaVA-Video-R1模型进行总结和决策
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化辩论主持Agent
        
        Args:
            name: Agent名称
            config: 配置参数
        """
        super().__init__(name, config)
        
        # 初始化TinyLLaVA模型 - 检查是否有共享模型实例
        model_path = config.get("model_path", "TinyLLaVA-Video-R1")
        if "shared_model" in config and config["shared_model"] is not None:
            self.logger.info("使用共享模型实例")
            self.model = config["shared_model"]
        else:
            self.load_model(model_path)
        
        # 辩论Agent注册表
        self.debate_agents = {}
        
        # 辩论轮数设置
        self.debate_rounds = config.get("debate_rounds", 2)
        
        # 辩论主题（人类动作意图识别）
        self.debate_topic = config.get("debate_topic", "人类动作意图识别")
        
        # 整合的辩论内容，用于保存到文件
        self.debate_content = []
        
        # 输出文件路径
        self.output_file = config.get("output_file", "/data/zhangyue/TinyLLaVA-Agent/agent_app/debate_output.txt")
        
        self.logger.info(f"辩论主持Agent '{name}' 已初始化，计划辩论轮数: {self.debate_rounds}")
    
    def _save_debate_to_file(self):
        """将辩论内容保存到文件"""
        try:
            dir_path = os.path.dirname(self.output_file)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for entry in self.debate_content:
                    f.write(entry + "\n\n")
            self.logger.info(f"辩论内容已保存到文件: {self.output_file}")
        except Exception as e:
            self.logger.error(f"保存辩论内容到文件失败: {str(e)}")
    
    def _add_to_debate_content(self, content: str):
        """添加内容到辩论记录"""
        self.debate_content.append(content)
    
    def load_model(self, model_path: str) -> None:
        """加载模型"""
        try:
            self.logger.info(f"正在加载模型: {model_path}")
            self.model = TinyLLaVA(model_path=model_path)
            self.logger.info("模型加载成功")
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def register_debate_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """注册一个辩论Agent"""
        self.debate_agents[agent_id] = agent
        self.logger.info(f"已注册辩论Agent: {agent_id}")
    
    async def start_debate(self, video_path: str) -> Dict[str, Any]:
        """启动辩论流程"""
        if not self.debate_agents:
            return {"success": False, "error": "没有注册的辩论Agent，无法开始辩论"}
        
        self.logger.info(f"开始针对视频 '{video_path}' 的辩论，共有 {len(self.debate_agents)} 个辩论Agent")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._add_to_debate_content(f"=== 辩论开始 ===\n时间: {timestamp}\n视频: {video_path}\n辩论Agent数量: {len(self.debate_agents)}\n辩论轮数: {self.debate_rounds}")
        
        perspectives_analyses = await self._collect_initial_analyses(video_path)
        if not perspectives_analyses.get("success", False):
            return perspectives_analyses
        
        debate_results = await self._conduct_debate(perspectives_analyses["analyses"])
        final_result = await self._summarize_debate(perspectives_analyses["analyses"], debate_results)
        
        self._save_debate_to_file()
        return final_result
    
    async def _collect_initial_analyses(self, video_path: str) -> Dict[str, Any]:
        """收集所有辩论Agent的初始分析结果"""
        self.logger.info("开始收集所有辩论Agent的初始分析结果")
        self._add_to_debate_content("\n=== 初始分析阶段 ===")
        
        analyses = {}
        failed_agents = []
        
        for agent_id, agent in self.debate_agents.items():
            self.logger.info(f"开始处理Agent '{agent_id}'的视频分析...")
            try:
                result = await agent.process({"video_path": video_path})
                if result.get("success", False):
                    analyses[agent_id] = result
                    self.logger.info(f"Agent '{agent_id}' 分析视频成功")
                    agent_perspective = result.get("agent_perspective", "未知视角")
                    content = result.get("content", "无内容")
                    self._add_to_debate_content(f"--- Agent: {agent_id} ({agent_perspective}) ---\n{content}")
                    
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        self.logger.info("已清理GPU缓存，为下一个Agent准备资源")
                else:
                    failed_agents.append(agent_id)
                    self.logger.error(f"Agent '{agent_id}' 分析视频失败: {result.get('error', '未知错误')}")
                    self._add_to_debate_content(f"--- Agent: {agent_id} ---\n分析失败: {result.get('error', '未知错误')}")
            except Exception as e:
                failed_agents.append(agent_id)
                self.logger.error(f"执行Agent '{agent_id}' 时出错: {str(e)}")
                self._add_to_debate_content(f"--- Agent: {agent_id} ---\n执行错误: {str(e)}")
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if not analyses:
            return {"success": False, "error": f"所有Agent分析视频失败: {', '.join(failed_agents)}"}
        
        self.logger.info(f"成功收集 {len(analyses)} 个Agent的分析结果，{len(failed_agents)} 个Agent失败")
        return {"success": True, "analyses": analyses, "failed_agents": failed_agents}
    
    async def _conduct_debate(self, initial_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """进行多轮辩论"""
        self.logger.info(f"开始进行 {self.debate_rounds} 轮辩论")
        
        debate_history = []
        current_arguments = {}
        
        for agent_id, analysis in initial_analyses.items():
            agent_perspective = analysis.get("agent_perspective", "未知视角")
            content = analysis.get("content", "")
            if content:
                current_arguments[agent_id] = {"agent_id": agent_id, "perspective": agent_perspective, "argument": content}
        
        active_agent_ids = list(current_arguments.keys())
        for round_num in range(1, self.debate_rounds + 1):
            self.logger.info(f"开始第 {round_num} 轮辩论")
            self._add_to_debate_content(f"\n=== 辩论轮次 {round_num} ===")
            
            round_results = {}
            for responder_id in active_agent_ids:
                if responder_id not in self.debate_agents:
                    self.logger.warning(f"Agent '{responder_id}' 不在注册列表中，跳过")
                    continue
                agent = self.debate_agents[responder_id]
                if responder_id not in current_arguments:
                    self.logger.warning(f"Agent '{responder_id}' 没有初始观点，跳过")
                    continue
                
                other_arguments = [arg for arg_id, arg in current_arguments.items() if arg_id != responder_id]
                if not other_arguments:
                    continue
                
                combined_arguments = "\n\n".join([f"From {arg['perspective']} perspective: {arg['argument']}" for arg in other_arguments])
                agent_perspective = current_arguments[responder_id]["perspective"]
                self._add_to_debate_content(f"--- 输入给 {responder_id} ({agent_perspective}) ---\n{combined_arguments}...")
                
                try:
                    self.logger.info(f"获取 {responder_id} 对其他Agent观点的回应")
                    response = await agent.process({"argument": combined_arguments})
                    if response.get("success", False) and "response" in response:
                        round_results[responder_id] = response
                        self._add_to_debate_content(f"--- 回应来自 {responder_id} ({agent_perspective}) ---\n{response['response']}")
                    else:
                        self.logger.error(f"Agent '{responder_id}' 回应失败")
                        self._add_to_debate_content(f"--- 回应来自 {responder_id} ({agent_perspective}) ---\n回应失败: {response.get('error', '未知错误')}")
                except Exception as e:
                    self.logger.error(f"Agent '{responder_id}' 在回应中发生错误: {str(e)}")
                    self._add_to_debate_content(f"--- 回应来自 {responder_id} ({agent_perspective}) ---\n执行错误: {str(e)}")
            
            for agent_id, response in round_results.items():
                if "response" in response:
                    current_arguments[agent_id]["argument"] = response["response"]
            
            debate_history.append({"round": round_num, "arguments": [arg for arg in current_arguments.values()]})
        
        self.logger.info("辩论完成")
        return {"success": True, "debate_history": debate_history, "final_arguments": current_arguments}
    
    async def _summarize_debate(self, initial_analyses: Dict[str, Dict[str, Any]], debate_results: Dict[str, Any]) -> Dict[str, Any]:
        """总结辩论结果并生成最终的意图识别结果"""
        self.logger.info("开始总结辩论结果")
        self._add_to_debate_content("\n=== 辩论总结 ===")
        
        perspectives_analyses_text = "".join(
            f"\n--- {analysis.get('agent_perspective', '未知视角').upper()} PERSPECTIVE ---\n{analysis.get('content', '')}\n"
            for agent_id, analysis in initial_analyses.items() if analysis.get("content")
        )
        
        debate_points_text = "".join(
            f"\n=== DEBATE ROUND {round_data.get('round', 0)} ===\n" + "".join(
                f"\n--- {arg.get('perspective', '未知视角').upper()} RESPONSE ---\n{arg.get('argument', '')}\n"
                for arg in round_data.get("arguments", [])
            )
            for round_data in debate_results.get("debate_history", [])
        )
        
        prompt = f"""You are a neutral debate moderator tasked with synthesizing multiple expert analyses of human behavior observed in a video.

        Provided below are:
        - Expert analyses from different perspectives:
        {perspectives_analyses_text}

        - Final viewpoints after {self.debate_rounds} rounds of debate:
        {debate_points_text}

        Your task is to:
        - Summarize the key points from each perspective.
        - Identify areas of agreement and disagreement.
        - Provide an integrated interpretation of the observed behavior.
        - Determine the most likely underlying intention behind the actions.

        Present your output **strictly** in the following JSON format:
        {{
        "integrated_analysis": "A concise synthesis combining all perspectives into a unified interpretation.",
        "final_intent_determination": "A clear conclusion stating the most probable intention behind the behavior.",
        "confidence_assessment": "A numerical confidence score between 0 and 1 representing the strength of the conclusion."
        }}

        Only output valid JSON. Avoid repeating information or including commentary outside the JSON structure.
        """

        
        try:
            self.logger.info("开始使用TinyLLaVA模型进行辩论总结...")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.model.analyze_text(prompt, max_length=2048))
            
            # 解析结果
            if result.get("success", False):
                summary_text = result.get("response", "")
                
                # 尝试从文本中提取JSON
                try:
                    # 处理各种可能的情况以提取JSON
                    json_str = None
                    
                    # 检查是否有代码块格式的JSON
                    if "```json" in summary_text and "```" in summary_text.split("```json", 1)[1]:
                        json_str = summary_text.split("```json", 1)[1].split("```", 1)[0].strip()
                    elif "```" in summary_text and "```" in summary_text.split("```", 1)[1]:
                        json_str = summary_text.split("```", 1)[1].split("```", 1)[0].strip()
                    # 检查是否有完整的JSON对象
                    elif "{" in summary_text and "}" in summary_text:
                        # 提取最外层的大括号内容
                        start_idx = summary_text.find("{")
                        end_idx = summary_text.rfind("}")
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = summary_text[start_idx:end_idx+1].strip()
                    
                    # 如果找到了可能的JSON字符串，尝试修复和解析
                    if json_str:
                        # 清理可能的格式问题
                        json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                        json_str = json_str.replace('\'', '"')  # 替换单引号为双引号
                        
                        # 移除可能导致错误的控制字符
                        json_str = ''.join(ch for ch in json_str if ord(ch) >= 32)
                        
                        # 尝试解析JSON
                        result_json = json.loads(json_str)
                    else:
                        # 没有找到JSON格式，创建一个基本结构
                        result_json = {
                            "integrated_analysis": summary_text, 
                            "final_intent_determination": "无法从模型响应中提取结构化意图",
                            "confidence_assessment": 0.5
                        }
                    
                    # 检查必要字段是否存在
                    required_fields = ["integrated_analysis", "final_intent_determination", "confidence_assessment"]
                    for field in required_fields:
                        if field not in result_json:
                            if field == "integrated_analysis":
                                result_json[field] = summary_text
                            elif field == "final_intent_determination":
                                result_json[field] = "无法确定具体意图"
                            elif field == "confidence_assessment":
                                result_json[field] = 0.5
                    
                    # 记录总结结果
                    self._add_to_debate_content(f"综合分析: {result_json.get('integrated_analysis', '')}")
                    self._add_to_debate_content(f"最终意图判断: {result_json.get('final_intent_determination', '')}")
                    self._add_to_debate_content(f"置信度: {result_json.get('confidence_assessment', 0.5)}")
                    
                    final_result = {
                        "success": True,
                        "content": result_json,
                        "raw_summary": summary_text
                    }
                except json.JSONDecodeError as json_error:
                    self.logger.warning(f"无法解析JSON结果: {str(json_error)}，返回原始文本")
                    
                    # 从文本中提取关键信息构建结构化结果
                    lines = summary_text.split('\n')
                    analysis = " ".join(lines)  
                    
                    # 记录原始输出
                    self._add_to_debate_content("无法解析为JSON格式，原始输出:")
                    self._add_to_debate_content(summary_text)
                    
                    final_result = {
                        "success": True,
                        "content": {
                            "integrated_analysis": analysis,
                            "final_intent_determination": "无法提取结构化意图，请查看整合分析",
                            "confidence_assessment": 0.5
                        },
                        "raw_summary": summary_text
                    }
            else:
                self.logger.error("模型分析失败")
                self._add_to_debate_content(f"模型分析失败: {result.get('error', '未知错误')}")
                final_result = {
                    "success": False,
                    "error": result.get("error", "未知错误")
                }
                
        except Exception as e:
            self.logger.error(f"总结辩论结果时出错: {str(e)}")
            self._add_to_debate_content(f"总结辩论结果时出错: {str(e)}")
            final_result = {
                "success": False,
                "error": f"总结辩论结果时出错: {str(e)}"
            }
        self._add_to_debate_content("\n=== 辩论结束 ===")
        return final_result
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据，启动辩论流程"""
        video_path = input_data.get("video_path")
        if not video_path:
            return {"success": False, "error": "缺少视频路径"}
        
        if "output_file" in input_data:
            self.output_file = input_data["output_file"]
            self.logger.info(f"设置自定义输出文件: {self.output_file}")
        
        return await self.start_debate(video_path)