"""
辅助工具函数
"""

import os
import base64
from typing import Dict, Any
from dotenv import load_dotenv


def load_dotenv_config() -> None:
    """
    加载环境变量配置
    """
    load_dotenv()


def encode_image_to_base64(image_path: str) -> str:
    """
    将图像编码为base64字符串
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        base64编码的图像字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_frames_from_video(video_path: str, output_dir: str, fps: int = 1) -> list:
    """
    从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        fps: 每秒提取的帧数
        
    Returns:
        提取的帧文件路径列表
    """
    import cv2
    import os
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    
    # 获取视频属性
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    # 提取帧
    frame_paths = []
    count = 0
    success = True
    
    while success:
        success, frame = video.read()
        
        if not success:
            break
            
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            
        count += 1
    
    video.release()
    
    return frame_paths


def format_response(success: bool, data: Any = None, error: str = None) -> Dict[str, Any]:
    """
    格式化API响应
    
    Args:
        success: 是否成功
        data: 响应数据
        error: 错误信息
        
    Returns:
        格式化的响应字典
    """
    return {
        "success": success,
        "data": data,
        "error": error
    } 