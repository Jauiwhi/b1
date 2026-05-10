from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import base64


class BaseModel(ABC):
    """模型基类，定义统一的接口"""

    def __init__(self, config: Dict):
        self.config = config
        self._vision_support = None

    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """发送对话请求，返回模型响应"""
        pass

    @abstractmethod
    def check_connection(self) -> bool:
        """检查模型连接是否有效"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """模型名称"""
        pass

    @property
    def supports_vision(self) -> bool:
        """模型是否支持视觉识别"""
        if self._vision_support is not None:
            return self._vision_support
        return self._detect_vision_support()

    def _detect_vision_support(self) -> bool:
        """通过测试判断模型是否支持视觉"""
        try:
            test_image = base64.b64encode(
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            ).decode('utf-8')

            response = self.chat([
                {"role": "user", "content": [
                    {"type": "text", "text": "回复YES"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image}"}}
                ]}
            ], max_tokens=10)
            self._vision_support = "YES" in response.upper()
        except Exception:
            self._vision_support = False
        return self._vision_support

    def count_tokens(self, text: str) -> int:
        """估算token数量（简单实现）"""
        return len(text) // 4

    def truncate_to_context(self, text: str, max_tokens: int = 32000) -> str:
        """将文本截断到指定token数量"""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def prepare_batch(self, contents: List[str], max_tokens: int = 32000) -> List[List[str]]:
        """将内容分批，确保每批不超过指定token数"""
        batches = []
        current_batch = []
        current_tokens = 0

        for content in contents:
            content_tokens = self.count_tokens(content)
            if current_tokens + content_tokens > max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(content)
            current_tokens += content_tokens

        if current_batch:
            batches.append(current_batch)

        return batches
