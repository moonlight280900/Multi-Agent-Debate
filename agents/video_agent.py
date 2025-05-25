"""
视频分析Agent
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
import json

from .base_agent import BaseAgent
from models.tinyllava import TinyLLaVA
from utils.helpers import extract_frames_from_video


class VideoAgent(BaseAgent):
    """
    视频分析Agent，负责处理视频内容并提取关键信息
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化视频分析Agent
        
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
            # 如果没有共享模型，则加载新的模型实例
            self.model = TinyLLaVA(model_path=model_path)
        
        # 设置提取帧的参数
        self.frames_per_second = config.get("frames_per_second", 1)
        self.temp_dir = config.get("temp_dir", "temp_frames")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 设置视频分析参数
        self.max_frames = config.get("max_frames", 16)
        self.direct_video_analysis = config.get("direct_video_analysis", True)
        
        # 协作模式配置
        self.collaborative_mode = config.get("collaborative_mode", False)
        self.collaborators = {}
        
        self.logger.info(f"VideoAgent已初始化，{'启用' if self.direct_video_analysis else '禁用'}直接视频分析，帧率:{self.frames_per_second}帧/秒")
    
    def register_collaborator(self, agent_id: str, agent: BaseAgent) -> None:
        """
        注册一个协作Agent
        
        Args:
            agent_id: Agent ID
            agent: Agent对象
        """
        self.collaborators[agent_id] = agent
        self.logger.info(f"已注册协作Agent: {agent_id}")
    
    async def _analyze_frame(self, frame_path: str) -> Dict[str, Any]:
        """
        分析单个视频帧
        
        Args:
            frame_path: 帧图像文件路径
            
        Returns:
            分析结果
        """
        # 使用英文提示词，因为TinyLLaVA使用英文输入
        prompt = """
        Analyze this video frame and describe:
        1. What actions is the person performing?
        2. What objects are they interacting with?
        3. What is the environment/context?
        4. What is the likely intention behind this action?
        
        Be as detailed as possible.
        """
        
        # 这里将同步函数包装为异步调用
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.model.analyze_video_frame(frame_path, prompt)
        )
        return result
    
    async def _analyze_video_direct(self, video_path: str) -> Dict[str, Any]:
        """
        直接分析整个视频（不提取帧）
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            分析结果
        """
        # 使用英文提示词
        prompt = """
        Analyze this video and describe in detail:
        1. What actions are being performed?
        2. What objects are visible and being interacted with?
        3. What is the environment/context?
        4. What is the likely intention behind the actions shown?
        5. Is there any notable event or change throughout the video?
        
        Be thorough in your analysis.Output the thinking process in <think> </think>
        """
        
        # 将同步函数包装为异步调用
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.model.analyze_video(video_path, prompt, num_frames=self.max_frames)
        )
        return result
    
    async def _process_frames(self, frame_paths: List[str]) -> List[Dict[str, Any]]:
        """
        处理多个视频帧
        
        Args:
            frame_paths: 帧图像文件路径列表
            
        Returns:
            处理结果列表
        """
        tasks = []
        for frame_path in frame_paths:
            task = asyncio.create_task(self._analyze_frame(frame_path))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # 将结果与帧路径匹配
        frame_results = []
        for i, result in enumerate(results):
            frame_results.append({
                "frame_path": frame_paths[i],
                "result": result
            })
        
        return frame_results
    
    async def _collaborate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        与其他Agent协作处理结果
        
        Args:
            data: 需要协作处理的数据
            
        Returns:
            协作处理后的结果
        """
        if not self.collaborators:
            self.logger.warning("没有注册的协作Agent，跳过协作处理")
            return data
        
        results = {}
        for agent_id, agent in self.collaborators.items():
            self.logger.info(f"与Agent '{agent_id}'协作处理数据")
            try:
                collab_result = await agent.process(data)
                results[agent_id] = collab_result
            except Exception as e:
                self.logger.error(f"与Agent '{agent_id}'协作失败: {str(e)}")
                results[agent_id] = {"success": False, "error": str(e)}
        
        # 将协作结果合并到原始数据中
        data["collaboration_results"] = results
        return data
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入视频数据
        
        Args:
            input_data: 包含视频路径的输入数据字典
            
        Returns:
            处理结果
        """
        video_path = input_data.get("video_path")
        if not video_path or not os.path.exists(video_path):
            return {
                "success": False,
                "error": f"视频文件不存在: {video_path}"
            }
        
        self.logger.info(f"开始处理视频: {video_path}")
        
        try:
            result = {}
            
            # 直接分析整个视频
            if self.direct_video_analysis:
                self.logger.info("使用直接视频分析模式")
                video_analysis = await self._analyze_video_direct(video_path)
                
                if not video_analysis.get("success", False):
                    self.logger.error(f"直接视频分析失败: {video_analysis.get('error')}")
                    return video_analysis
                
                result = {
                    "success": True,
                    "video_path": video_path,
                    "analysis_type": "direct_video",
                    "content": video_analysis.get("content"),
                    "num_frames": video_analysis.get("num_frames"),
                    "message": "成功直接分析视频"
                }
            else:
                # 传统方式：提取帧然后逐帧分析
                self.logger.info("使用逐帧分析模式")
                # 提取视频帧
                frame_paths = extract_frames_from_video(
                    video_path, 
                    self.temp_dir, 
                    fps=self.frames_per_second
                )
                
                self.logger.info(f"已提取{len(frame_paths)}帧")
                
                # 分析帧
                frame_results = await self._process_frames(frame_paths)
                
                result = {
                    "success": True,
                    "video_path": video_path,
                    "analysis_type": "frame_by_frame",
                    "frame_count": len(frame_paths),
                    "frame_results": frame_results,
                    "message": f"成功处理视频，共{len(frame_paths)}帧"
                }
            
            # 更新状态
            self.update_state(result)
            
            # 如果启用了协作模式，则与其他Agent协作处理结果
            if self.collaborative_mode and self.collaborators:
                self.logger.info("启动协作处理模式")
                result = await self._collaborate(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理视频时出错: {str(e)}")
            return {
                "success": False,
                "error": f"处理视频时出错: {str(e)}"
            } 