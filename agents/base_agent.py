"""
基础Agent类定义
"""

import time
from typing import Dict, Any, List, Optional, Callable
import logging
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BaseAgent(ABC):
    """
    基础Agent抽象类
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化基础Agent
        
        Args:
            name: Agent名称
            config: 配置参数
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"Agent:{name}")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 60)
        self.state = {}
        self.logger.info(f"Agent {name} 已初始化")
    
    @abstractmethod
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        处理输入数据的抽象方法
        
        Args:
            input_data: 输入数据
            
        Returns:
            处理结果
        """
        pass
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        带重试的执行函数
        
        Args:
            func: 要执行的函数
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            函数执行结果
        """
        retries = 0
        last_exception = None
        
        while retries < self.max_retries:
            try:
                start_time = time.time()
                result = await func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                
                self.logger.info(f"执行成功，耗时: {elapsed_time:.2f}秒")
                return result
                
            except Exception as e:
                retries += 1
                last_exception = e
                self.logger.warning(f"执行失败 ({retries}/{self.max_retries}): {str(e)}")
                time.sleep(1)  # 简单的指数退避策略
        
        # 所有重试都失败了
        self.logger.error(f"执行失败，已达到最大重试次数: {last_exception}")
        raise last_exception
    
    def update_state(self, state_update: Dict[str, Any]) -> None:
        """
        更新Agent状态
        
        Args:
            state_update: 状态更新字典
        """
        self.state.update(state_update)
        self.logger.debug(f"状态已更新: {state_update}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取Agent状态
        
        Returns:
            当前状态字典
        """
        return self.state.copy()
    
    def reset(self) -> None:
        """
        重置Agent状态
        """
        self.state = {}
        self.logger.info("Agent状态已重置") 