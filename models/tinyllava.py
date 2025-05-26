"""
TinyLLaVA-Video-R1模型接口 - 本地模型版本
"""

import json
import base64
from typing import Dict, List, Any, Optional
import os
import sys
import torch
from PIL import Image
import requests
import importlib.util
import importlib
import subprocess
from pathlib import Path
from io import BytesIO
import numpy as np
import cv2
import torch

class TinyLLaVA:
    """TinyLLaVA本地模型接口"""
    
    def __init__(self, model_path: str = None, gpu_id: int = None):
        """
        初始化TinyLLaVA接口
        
        Args:
            model_path: 本地模型路径，如果为None则使用默认路径
            gpu_id: 指定使用的GPU ID，默认为None时使用系统默认设备
        """
        self.model_path = model_path or "TinyLLaVA-Video-R1"
        
        # 设置使用的GPU设备
        if gpu_id is not None:
            self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            # 设置当前CUDA设备
            torch.cuda.set_device(gpu_id)
            print(f"使用GPU设备: {self.device} (ID: {gpu_id})")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.image_processor = None
        
        # 确保TinyLLaVA-Video-R1路径在系统路径中
        tinyllava_dir = Path(self.model_path) / "TinyLLaVA-Video-R1"
        if tinyllava_dir.exists() and str(tinyllava_dir) not in sys.path:
            sys.path.append(str(tinyllava_dir))
        elif not tinyllava_dir.exists() and str(self.model_path) not in sys.path:
            sys.path.append(str(self.model_path))
            
        self._load_model()
        
    def _find_model_path(self, path: str) -> str:
        """
        查找有效的模型路径
        
        Args:
            path: 用户提供的路径
            
        Returns:
            正确的模型路径
        """
        path = Path(path)
        
        # 检查是否是有效的模型路径
        model_file_paths = [
            path / "config.json",
            path / "pytorch_model.bin",
            path / "pytorch_model-00001-of-00002.bin"
        ]
        
        # 如果提供的路径直接是模型路径
        if any(p.exists() for p in model_file_paths):
            print(f"找到模型文件: {path}")
            return str(path)
            
        # 检查TinyLLaVA-Video-R1/TinyLLaVA-Video-R1格式
        if (path / "TinyLLaVA-Video-R1").exists():
            nested_path = path / "TinyLLaVA-Video-R1"
            nested_model_paths = [
                nested_path / "config.json",
                nested_path / "pytorch_model.bin",
                nested_path / "pytorch_model-00001-of-00002.bin"
            ]
            
            if any(p.exists() for p in nested_model_paths):
                print(f"在子目录找到模型文件: {nested_path}")
                return str(nested_path)
                
        # 没有找到模型文件，返回原始路径
        print(f"未找到模型文件，使用原始路径: {path}")
        return str(path)
    
    def _load_model(self):
        """加载本地TinyLLaVA模型"""
        try:
            # 导入TinyLLaVA-Video-R1中的必要模块
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
            
            # 尝试导入TinyLLaVA-Video-R1的eval相关模块
            try:
                # 先尝试直接导入
                from tinyllava.eval.run_tiny_llava import load_pretrained_model, disable_torch_init
                from tinyllava.data import TextPreprocess, ImagePreprocess, VideoPreprocess
                from tinyllava.utils.message import Message
            except ImportError:
                print("直接导入失败，尝试使用importlib动态导入...")
                # 使用动态导入方式
                tinyllava_path = Path(self.model_path) / "TinyLLaVA-Video-R1"
                
                # 导入run_tiny_llava模块
                spec = importlib.util.spec_from_file_location(
                    "run_tiny_llava", 
                    str(tinyllava_path / "tinyllava" / "eval" / "run_tiny_llava.py")
                )
                run_tiny_llava = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(run_tiny_llava)
                
                # 导入data模块
                spec_data = importlib.util.spec_from_file_location(
                    "data",
                    str(tinyllava_path / "tinyllava" / "data" / "__init__.py")
                )
                data_module = importlib.util.module_from_spec(spec_data)
                spec_data.loader.exec_module(data_module)
                
                # 导入message模块
                spec_message = importlib.util.spec_from_file_location(
                    "message",
                    str(tinyllava_path / "tinyllava" / "utils" / "message.py")
                )
                message_module = importlib.util.module_from_spec(spec_message)
                spec_message.loader.exec_module(message_module)
                
                load_pretrained_model = run_tiny_llava.load_pretrained_model
                disable_torch_init = run_tiny_llava.disable_torch_init
                TextPreprocess = data_module.TextPreprocess
                ImagePreprocess = data_module.ImagePreprocess
                VideoPreprocess = data_module.VideoPreprocess
                Message = message_module.Message
            
            # 禁用torch的初始化，这是TinyLLaVA-Video-R1中的做法
            disable_torch_init()
            
            # 加载预训练模型
            self.model, self.tokenizer, self.image_processor, self.context_len = load_pretrained_model(self.model_path)
            # 将模型移到GPU上
            self.model.to(self.device)
            
            # 初始化处理器
            self.text_processor = TextPreprocess(self.tokenizer, "qwen2_base")  # 默认使用qwen2_base作为conv_mode
            self.data_args = self.model.config
            self.image_preprocess = ImagePreprocess(self.image_processor, self.data_args)
            self.video_preprocess = VideoPreprocess(self.image_processor, self.data_args)
            
            print(f"成功加载TinyLLaVA模型从: {self.model_path}")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise
    
    def analyze_video_frame(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        分析单个视频帧
        
        Args:
            image_path: 图像文件路径
            prompt: 提示词
            
        Returns:
            模型响应
        """
        try:
            # 加载图像
            if image_path.startswith("http") or image_path.startswith("https"):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
            
            # 添加DEFAULT_IMAGE_TOKEN到提示词
            from tinyllava.utils import DEFAULT_IMAGE_TOKEN
            prompt_with_token = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            
            # 导入必要的类
            try:
                from tinyllava.utils.message import Message
            except ImportError:
                # 如果直接导入失败，尝试动态导入
                tinyllava_path = Path(self.model_path) / "TinyLLaVA-Video-R1"
                if not tinyllava_path.exists():
                    tinyllava_path = Path(self.model_path)
                
                spec_message = importlib.util.spec_from_file_location(
                    "message",
                    str(tinyllava_path / "tinyllava" / "utils" / "message.py")
                )
                message_module = importlib.util.module_from_spec(spec_message)
                spec_message.loader.exec_module(message_module)
                Message = message_module.Message
            
            # 使用Message和TextPreprocess处理文本
            msg = Message()
            msg.add_message(prompt_with_token)
            result = self.text_processor(msg.messages, mode='eval')
            
            input_ids = result['input_ids']
            input_ids = input_ids.unsqueeze(0).to(self.device)
            
            # 处理图像
            images_tensor = self.image_preprocess(image)
            images_tensor = images_tensor.unsqueeze(0).half().to(self.device)
            
            # 获取停止生成的标记
            stop_str = self.text_processor.template.separator.apply()[1]
            keywords = [stop_str]
            
            # 导入停止标准
            from tinyllava.eval.run_tiny_llava import KeywordsStoppingCriteria
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            
            # 生成回答，使用更严格的参数限制
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    video=None,
                    do_sample=True,
                    temperature=0.5,  # 稍微提高温度以减少重复
                    top_p=0.9,        # 增加多样性
                    num_beams=3,      # 使用beam search增加生成质量
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=512,  # 减少最大生成长度
                    use_cache=True,
                    repetition_penalty=2.0,  # 添加重复惩罚
                    stopping_criteria=[stopping_criteria],
                )
            
            # 解码输出
            outputs = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            
            
            return {
                "success": True,
                "content": outputs
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_video(self, video_path: str, prompt: str, num_frames: int = 16) -> Dict[str, Any]:
        """
        分析整个视频
        
        Args:
            video_path: 视频文件路径
            prompt: 提示词
            num_frames: 要提取的帧数
            
        Returns:
            模型响应
        """
        try:
            # 检查是否为纯文本分析模式
            if video_path is None:
                return self.analyze_text(prompt)
                
            # 添加DEFAULT_IMAGE_TOKEN到提示词
            from tinyllava.utils import DEFAULT_IMAGE_TOKEN
            prompt_with_token = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            
            # 导入必要的类
            try:
                from tinyllava.utils.message import Message
            except ImportError:
                # 如果直接导入失败，尝试动态导入
                tinyllava_path = Path(self.model_path) / "TinyLLaVA-Video-R1"
                if not tinyllava_path.exists():
                    tinyllava_path = Path(self.model_path)
                
                spec_message = importlib.util.spec_from_file_location(
                    "message",
                    str(tinyllava_path / "tinyllava" / "utils" / "message.py")
                )
                message_module = importlib.util.module_from_spec(spec_message)
                spec_message.loader.exec_module(message_module)
                Message = message_module.Message
            
            # 使用Message和TextPreprocess处理文本
            msg = Message()
            msg.add_message(prompt_with_token)
            result = self.text_processor(msg.messages, mode='eval')
            
            input_ids = result['input_ids']
            print(f"""msg is {msg.messages} \n||| result is {result}""")
            input_ids = input_ids.unsqueeze(0).to(self.device)
            
            # 使用更简单的方法处理视频 - 直接用cv2提取帧
            print(f"开始处理视频: {video_path}")
            
            # 主动清理GPU内存
            torch.cuda.empty_cache()
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频: {video_path}")
                return {
                    "success": False,
                    "error": f"无法打开视频: {video_path}"
                }
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"视频信息: 总帧数={total_frames}, FPS={fps}, 时长={duration}秒")
            
            # 减少帧数以避免内存溢出问题
            # 视频网格是模型处理的内存瓶颈点
            reduced_frames = min(16, total_frames)  # 进一步减少到最多6帧
            print(f"为避免内存溢出，减少处理帧数至: {reduced_frames}")
                
            # 计算均匀间隔的帧索引
            if total_frames <= reduced_frames:
                # 如果总帧数不足，则全部使用
                frame_indices = list(range(total_frames))
            else:
                # 计算均匀间隔
                step = total_frames / reduced_frames
                frame_indices = [int(i * step) for i in range(reduced_frames)]
            
            print(f"计划提取的帧索引: {frame_indices}")
            
            # 提取指定帧
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # 转换颜色空间从BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 将numpy数组转换为PIL图像
                    img = Image.fromarray(frame_rgb)
                    frames.append(img)
                else:
                    print(f"无法读取帧 {frame_idx}")
            
            # 释放视频资源
            cap.release()
            
            if not frames:
                print("没有成功提取任何帧")
                return {
                    "success": False,
                    "error": "无法提取视频帧"
                }
                
            print(f"成功提取 {len(frames)} 帧")
            
            # 处理提取的帧 - 考虑降低分辨率以减少内存使用
            processed_frames = []
            for frame in frames:
                try:
                    # 可选: 调整图像大小以减少内存使用
                    # frame = frame.resize((224, 224), Image.LANCZOS) 
                    # 使用图像预处理 (而不是视频预处理)
                    processed_frame = self.image_preprocess(frame)
                    processed_frames.append(processed_frame)
                except Exception as e:
                    print(f"处理帧时出错: {str(e)}")
                    # 继续处理其他帧
            
            if not processed_frames:
                print("没有成功处理任何帧")
                return {
                    "success": False,
                    "error": "所有帧处理失败"
                }
            
            # 将处理后的帧堆叠成视频张量
            video_tensor = torch.stack(processed_frames)
            video_tensor = video_tensor.unsqueeze(0).to(self.device)  # 添加批次维度
            
            print(f"视频张量形状: {video_tensor.shape}")
            
            # 主动清理内存以减少内存溢出风险
            torch.cuda.empty_cache()
            
            # 获取停止生成的标记
            stop_str = self.text_processor.template.separator.apply()[1]
            keywords = [stop_str]
            
            # 停止标准
            from tinyllava.eval.run_tiny_llava import KeywordsStoppingCriteria
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            
            # 生成回答
            print("开始生成回答...")
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=None,
                    video=video_tensor,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    num_beams=3,  
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=512,  # 增加生成长度以获得更完整的回答
                    use_cache=True,
                    repetition_penalty=2.0,  # 提高重复惩罚以减少重复
                    stopping_criteria=[stopping_criteria],
                )
            
            # 解码输出
            outputs = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            print(f"""content is : {outputs}""")
            outputs = outputs.strip()
            print(f"""content1 is : {outputs}""")
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            print(f"""content2 is : {outputs}""")
            return {
                "success": True,
                "content": outputs,
                "num_frames": len(processed_frames)
            }
            
        except torch.cuda.OutOfMemoryError as e:
            # 专门处理CUDA内存溢出错误
            print(f"CUDA内存溢出错误: {str(e)}")
            # 清理GPU内存
            torch.cuda.empty_cache()
            # 尝试使用纯文本分析作为回退方案
            print("尝试回退到纯文本分析...")
            try:
                return self.analyze_text(prompt)
            except Exception as text_error:
                print(f"纯文本分析也失败: {str(text_error)}")
                return {
                    "success": False,
                    "error": f"视频处理内存溢出，文本回退也失败: {str(e)}"
                }
        except Exception as e:
            print(f"视频处理错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 尝试使用纯文本分析作为回退方案
            try:
                return self.analyze_text(prompt)
            except:
                return {
                    "success": False,
                    "error": f"视频处理错误: {str(e)}"
                }
    
    def analyze_text(self, prompt: str, max_length: int = 512) -> Dict[str, Any]:
        """
        纯文本分析，不需要视频输入
        
        Args:
            prompt: 提示词
            max_length: 生成文本的最大长度，默认为512
            
        Returns:
            模型响应
        """
        try:
            print(f"analyze_text方法开始处理文本，提示词长度: {len(prompt)}")
            
            # 主动清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("已清理GPU缓存，准备进行文本分析")
                
            # 不添加图像标记，直接使用文本
            # 导入必要的类
            try:
                from tinyllava.utils.message import Message
                print("成功导入Message类")
            except ImportError as import_error:
                print(f"直接导入Message失败: {str(import_error)}，尝试动态导入")
                # 如果直接导入失败，尝试动态导入
                tinyllava_path = Path(self.model_path) / "TinyLLaVA-Video-R1"
                if not tinyllava_path.exists():
                    tinyllava_path = Path(self.model_path)
                
                print(f"动态导入路径: {tinyllava_path}")
                spec_message = importlib.util.spec_from_file_location(
                    "message",
                    str(tinyllava_path / "tinyllava" / "utils" / "message.py")
                )
                message_module = importlib.util.module_from_spec(spec_message)
                spec_message.loader.exec_module(message_module)
                Message = message_module.Message
                print("动态导入Message成功")
            
            # 使用Message和TextPreprocess处理文本
            print("创建Message实例并添加消息...")
            msg = Message()
            msg.add_message(prompt)
            print("使用text_processor处理消息...")
            result = self.text_processor(msg.messages, mode='eval')
            
            input_ids = result['input_ids']
            input_ids = input_ids.unsqueeze(0).to(self.device)
            print(f"输入ID张量形状: {input_ids.shape}")
            
            # 获取停止生成的标记
            stop_str = self.text_processor.template.separator.apply()[1]
            keywords = [stop_str]
            print(f"设置停止标志: {stop_str}")
            
            # 导入停止标准
            from tinyllava.eval.run_tiny_llava import KeywordsStoppingCriteria
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            
            # 再次清理内存，确保有足够空间进行生成
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 使用更保守的参数配置，降低内存使用
            # 生成回答
            print("开始生成回答...")
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=None,
                    video=None,  # 不提供视频输入
                    do_sample=True,
                    temperature=0.7,
                    top_p=None,
                    num_beams=3,  # 使用beam=1降低内存使用
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=max_length,  # 使用传入的max_length参数
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            
            # 解码输出
            print("解码输出...")
            outputs = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("已清理GPU缓存，完成文本处理")
            
            print(f"处理完成，输出长度: {len(outputs)}")
            
            return {
                "success": True,
                "response": outputs
            }
            
        except Exception as e:
            print(f"analyze_text方法出错: {str(e)}")
            # 发生异常时也清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "success": False,
                "error": str(e)
            }
    
    
    def analyze_video_frames(self, frame_paths: List[str], prompt: str) -> List[Dict[str, Any]]:
        """
        批量分析视频帧
        
        Args:
            frame_paths: 帧图像文件路径列表
            prompt: 提示词
            
        Returns:
            分析结果列表
        """
        results = []
        
        for frame_path in frame_paths:
            result = self.analyze_video_frame(frame_path, prompt)
            results.append({
                "frame_path": frame_path,
                "result": result
            })
            
        return results 