from abc import ABC, abstractmethod
from typing import Dict, Optional
from pathlib import Path


class BaseProcessor(ABC):
    """文件处理器基类"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """从文件中提取文本内容"""
        pass

    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict:
        """提取文件元数据"""
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> list:
        """支持的文件扩展名"""
        pass

    def validate_file(self, file_path: str) -> bool:
        """验证文件是否有效"""
        path = Path(file_path)
        if not path.exists():
            return False
        ext = path.suffix.lower()
        return ext in self.supported_extensions

    def get_file_info(self, file_path: str) -> Dict:
        """获取文件基本信息"""
        path = Path(file_path)
        return {
            'path': file_path,
            'name': path.name,
            'extension': path.suffix.lower(),
            'size': path.stat().st_size if path.exists() else 0
        }
