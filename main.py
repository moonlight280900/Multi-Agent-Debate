"""
多Agent视频动作意图识别系统主程序
"""

import os
import asyncio
import argparse
import logging
from typing import Dict, Any, List
import gc
import torch
from config.settings import Settings
from agents.debate_agent import DebateAgent
from agents.debate_moderator_agent import DebateModeratorAgent
from utils.helpers import load_dotenv_config
from models.tinyllava import TinyLLaVA

# 全局变量，用于存储共享的模型实例
SHARED_MODEL = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")


def get_or_create_shared_model(model_path: str = None, gpu_id: int = None) -> TinyLLaVA:
    """
    获取或创建共享的模型实例
    
    Args:
        model_path: 模型路径
        gpu_id: 指定使用的GPU ID
        
    Returns:
        共享的模型实例
    """
    global SHARED_MODEL
    
    if SHARED_MODEL is None:
        logger.info(f"创建共享TinyLLaVA模型实例: {model_path}, GPU ID: {gpu_id}")
        
        # 创建新的模型实例
        SHARED_MODEL = TinyLLaVA(model_path=model_path, gpu_id=gpu_id)
    
    return SHARED_MODEL

async def process_video_debate(video_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用多个辩论Agent和辩论主持Agent对视频中的人类动作意图进行辩论和识别
    
    Args:
        video_path: 视频文件路径
        config: 配置参数
        
    Returns:
        辩论结果和最终意图识别结果
    """
    if not os.path.exists(video_path):
        return {
            "success": False,
            "error": f"视频文件不存在: {video_path}"
        }
    
    logger.info(f"开始使用辩论Agent对视频进行多角度动作意图分析: {video_path}")
    
    # 获取或创建共享模型实例
    try:
        model_path = config.get("model_path", "TinyLLaVA-Video-R1")
        gpu_id = config.get("gpu_id", 0)
        shared_model = get_or_create_shared_model(model_path, gpu_id)
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return {
            "success": False,
            "error": f"模型加载失败: {str(e)}"
        }
    
    # 基础配置
    base_config = {
        "model_path": model_path,
        "max_frames": config.get("max_frames", 16),
        "max_retries": config.get("max_retries", 3),
        "timeout": config.get("agent_timeout", 60),
        "shared_model": shared_model  # 使用共享模型实例
    }
    
    # 创建多个具有不同视角的辩论Agent
    debate_agents = {}
    
    # 1. 行为动作专注Agent
    action_agent = DebateAgent("行为动作专家", "action_focused", base_config.copy())
    debate_agents["action"] = action_agent
    
    # 2. 意图专注Agent
    intention_agent = DebateAgent("意图动机专家", "intention_focused", base_config.copy())
    debate_agents["intention"] = intention_agent
    
    # 3. 环境情境专注Agent
    context_agent = DebateAgent("环境情境专家", "context_focused", base_config.copy())
    debate_agents["context"] = context_agent
    
    # 4. 情绪专注Agent
    emotion_agent = DebateAgent("情绪情感专家", "emotional_focused", base_config.copy())
    debate_agents["emotion"] = emotion_agent
    
    # 5. 社会因素专注Agent
    social_agent = DebateAgent("社会关系专家", "social_focused", base_config.copy())
    debate_agents["social"] = social_agent
    
    # 创建辩论主持Agent
    moderator_agent = DebateModeratorAgent("辩论主持Agent", {
        "model_path": model_path,
        "shared_model": shared_model,  # 使用共享模型实例
        "debate_rounds": config.get("debate_rounds", 2),
        "max_retries": config.get("max_retries", 3),
        "timeout": config.get("agent_timeout", 120),
        "output_file": config.get("debate_output_file", "debate_output.txt")  # 设置输出文件
    })
    
    # 注册所有辩论Agent到主持Agent
    for agent_id, agent in debate_agents.items():
        moderator_agent.register_debate_agent(agent_id, agent)
    
    # 启动辩论流程
    try:
        logger.info(f"启动辩论流程，共有 {len(debate_agents)} 个不同视角的辩论Agent")
        result = await moderator_agent.process({"video_path": video_path})
        return result
    except Exception as e:
        logger.error(f"辩论流程执行出错: {str(e)}")
        return {
            "success": False,
            "error": f"辩论流程执行出错: {str(e)}"
        }


async def process_video(video_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理视频并识别动作意图
    
    Args:
        video_path: 视频文件路径
        config: 配置参数
        
    Returns:
        处理结果
    """
    if not os.path.exists(video_path):
        return {
            "success": False,
            "error": f"视频文件不存在: {video_path}"
        }
    
    # 根据配置选择处理方式
    if config.get("use_debate_agents", False):
        # 使用辩论Agent方式
        logger.info("使用多视角辩论方式分析视频")
        return await process_video_debate(video_path, config)
    elif config.get("direct_model", False):
        # 直接使用模型分析
        logger.info("process video error")
        

def main():
    """
    主函数
    """
    # 加载环境变量配置
    load_dotenv_config()
    
    # 全局变量声明
    global SHARED_MODEL
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="TinyLLaVA-Agent: 多Agent视频动作意图识别系统")
    
    # 视频相关参数
    parser.add_argument("--video", "-v", type=str, help="要分析的视频文件路径")
    parser.add_argument("--fps", type=int, default=1, help="每秒提取的帧数（默认：1）")
    parser.add_argument("--max-frames", type=int, default=16, help="视频分析最大帧数（默认：16）")
    
    # 模型相关参数
    parser.add_argument("--model-path", "-m", type=str, help="TinyLLaVA模型路径")
    parser.add_argument("--gpu", type=int, default=1, help="指定使用的GPU ID（默认：1）")
    parser.add_argument("--prompt", type=str, help="自定义分析提示词")
    
    # 处理模式参数
    parser.add_argument("--multi-agent", action="store_true", help="启用多Agent协作模式")
    parser.add_argument("--direct-model", action="store_true", help="直接使用模型分析，跳过Agent协作")
    parser.add_argument("--direct-video", action="store_true", help="直接分析整个视频而非逐帧分析")
    parser.add_argument("--debate-agents", action="store_true", help="使用多个辩论Agent进行分析")
    parser.add_argument("--debate-rounds", type=int, default=2, help="辩论轮数（默认：2）")
    
    # API相关参数
    parser.add_argument("--api", "-a", action="store_true", help="启动API服务模式")
    
    # 输出相关参数
    parser.add_argument("--output", "-o", type=str, help="结果输出文件路径")
    parser.add_argument("--debate-output", type=str, default="debate_output.txt", 
                        help="辩论输出文件路径（默认：debate_output.txt）")
    
    args = parser.parse_args()
    
    # 加载配置
    settings = Settings()
    
    # 准备配置参数
    config = {
        "model_path": args.model_path or settings.model_path,
        "frames_per_second": args.fps,
        "max_frames": args.max_frames,
        "direct_video_analysis": args.direct_video,
        "gpu_id": args.gpu,
        "prompt": args.prompt,
        "openai_api_key": settings.openai_api_key,
        "openai_base_url": settings.openai_base_url,
        "direct_model": args.direct_model,
        "use_debate_agents": args.debate_agents,
        "debate_rounds": args.debate_rounds,
        "debate_output_file": args.debate_output  # 添加辩论输出文件配置
    }
    
        
    if args.video:
        # 视频处理模式
        logger.info(f"处理视频: {args.video}")
        
        # 创建异步事件循环
        loop = asyncio.get_event_loop()
        
        try:
            # 处理视频
            result = loop.run_until_complete(process_video(args.video, config))
            
            # 输出结果
            if result.get("success", False):
                logger.info("处理成功")
                
                # 如果指定了输出文件，将结果保存到文件
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        import json
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"结果已保存到: {args.output}")
                else:
                    # 直接打印结果
                    print("\n结果:")
                    print("=" * 80)
                    for key, value in result.items():
                        if key != "success":
                            print(f"\n## {key} ##")
                            print("-" * 40)
                            if isinstance(value, dict) and "content" in value:
                                print(value["content"])
                            elif isinstance(value, str):
                                print(value)
                            else:
                                print(str(value))
                    print("=" * 80)
            else:
                logger.error(f"处理失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            logger.error(f"处理视频时出错: {str(e)}")
            
        finally:
            # 清理资源
            if SHARED_MODEL is not None:
                logger.info("释放模型资源")
                del SHARED_MODEL
                gc.collect()
                torch.cuda.empty_cache()
    
    else:
        # 无参数模式，打印帮助信息
        parser.print_help()


if __name__ == "__main__":
    main() 