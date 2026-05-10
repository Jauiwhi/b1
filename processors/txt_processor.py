from typing import Dict, List
from pathlib import Path
from .base import BaseProcessor


class TxtProcessor(BaseProcessor):
    """纯文本文件处理器"""

    @property
    def supported_extensions(self) -> List[str]:
        return ['.txt']

    def extract_text(self, file_path: str) -> str:
        """读取文本文件内容"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
        for encoding in encodings:
            try:
                return Path(file_path).read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return Path(file_path).read_text(encoding='utf-8', errors='ignore')

    def extract_metadata(self, file_path: str) -> Dict:
        """提取文本文件元数据"""
        info = self.get_file_info(file_path)
        metadata = {
            **info,
            'type': 'text',
            'encoding': self._detect_encoding(file_path),
            'line_count': 0,
            'word_count': 0
        }

        try:
            text = self.extract_text(file_path)
            metadata['line_count'] = len(text.splitlines())
            metadata['word_count'] = len(text.split())
        except Exception:
            pass

        return metadata

    def _detect_encoding(self, file_path: str) -> str:
        """检测文件编码"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
        for encoding in encodings:
            try:
                Path(file_path).read_text(encoding=encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        return 'unknown'
