"""
视频动作意图识别API服务
"""

import os
import tempfile
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import uuid

from config.settings import Settings
from agents.video_agent import VideoAgent
from agents.intent_agent import IntentAgent
from utils.helpers import format_response, load_dotenv_config

# 加载环境变量
load_dotenv_config()

# 创建应用实例
app = FastAPI(
    title="视频动作意图识别系统",
    description="基于多Agent架构的视频动作意图识别系统API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# 加载配置
settings = Settings()

# 初始化Agent - 只创建IntentAgent，VideoAgent将在需要时创建
intent_agent = IntentAgent("意图识别Agent", {
    "openai_api_key": settings.openai_api_key,
    "action_intents": settings.action_intents,
    "max_retries": settings.max_retries,
    "timeout": settings.agent_timeout
})

# 任务存储
task_results = {}


@app.get("/")
async def root():
    """API根路径"""
    return {"message": "视频动作意图识别系统API"}


@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    frames_per_second: Optional[int] = Form(1),
    model_path: Optional[str] = Form(None)
):
    """
    分析视频并识别动作意图
    
    Args:
        video: 要分析的视频文件
        frames_per_second: 每秒提取的帧数
        model_path: 可选的TinyLLaVA模型路径
        
    Returns:
        任务ID
    """
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 保存上传的视频
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, f"{task_id}_{video.filename}")
        
        # 写入视频文件
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        # 添加后台任务
        background_tasks.add_task(
            process_video_task,
            task_id=task_id,
            video_path=video_path,
            frames_per_second=frames_per_second,
            model_path=model_path
        )
        
        return format_response(
            success=True,
            data={
                "task_id": task_id,
                "message": "视频分析任务已提交"
            }
        )
        
    except Exception as e:
        logger.error(f"处理视频上传时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理视频时出错: {str(e)}")


@app.get("/tasks/{task_id}")
async def get_task_result(task_id: str):
    """
    获取任务结果
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务结果
    """
    if task_id not in task_results:
        return format_response(
            success=False,
            error=f"找不到任务ID: {task_id}"
        )
    
    return format_response(
        success=True,
        data=task_results[task_id]
    )


async def process_video_task(
    task_id: str, 
    video_path: str, 
    frames_per_second: int = 1,
    model_path: str = None
):
    """
    处理视频任务
    
    Args:
        task_id: 任务ID
        video_path: 视频文件路径
        frames_per_second: 每秒提取的帧数
        model_path: 可选的TinyLLaVA模型路径
    """
    try:
        # 更新任务状态
        task_results[task_id] = {
            "status": "processing",
            "message": "正在分析视频..."
        }
        
        # 使用提供的模型路径或默认路径
        model_path = model_path or settings.model_path
        
        # 创建并配置视频Agent
        video_agent = VideoAgent("视频分析Agent", {
            "model_path": model_path,
            "frames_per_second": frames_per_second,
            "temp_dir": "temp_frames",
            "max_retries": settings.max_retries,
            "timeout": settings.agent_timeout
        })
        
        # 1. 使用VideoAgent处理视频
        video_result = await video_agent.process({"video_path": video_path})
        
        if not video_result.get("success", False):
            task_results[task_id] = {
                "status": "failed",
                "message": video_result.get("error", "视频处理失败")
            }
            return
        
        # 更新任务状态
        task_results[task_id] = {
            "status": "processing",
            "message": "视频分析完成，正在识别动作意图...",
            "video_result": {
                "frame_count": video_result.get("frame_count", 0)
            }
        }
        
        # 2. 使用IntentAgent识别意图
        intent_result = await intent_agent.process(video_result)
        
        if not intent_result.get("success", False):
            task_results[task_id] = {
                "status": "failed",
                "message": intent_result.get("error", "意图识别失败"),
                "video_result": {
                    "frame_count": video_result.get("frame_count", 0)
                }
            }
            return
        
        # 3. 更新任务完成状态
        task_results[task_id] = {
            "status": "completed",
            "message": "分析完成",
            "video_result": {
                "frame_count": video_result.get("frame_count", 0)
            },
            "intent_result": intent_result.get("intent_result") or {
                "raw_response": intent_result.get("raw_response", "")
            }
        }
        
    except Exception as e:
        logger.error(f"处理视频任务时出错: {str(e)}")
        task_results[task_id] = {
            "status": "failed",
            "message": f"任务处理过程中出错: {str(e)}"
        }


def start():
    """启动API服务"""
    uvicorn.run(
        "api.server:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )


if __name__ == "__main__":
    start() 