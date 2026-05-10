from typing import Dict, Optional
from pathlib import Path
from .base import BaseProcessor
from .txt_processor import TxtProcessor
from .pdf_processor import PdfProcessor
from .docx_processor import DocxProcessor
from .xlsx_processor import XlsxProcessor
from .image_processor import ImageProcessor


class ProcessorFactory:
    """文件处理器工厂"""

    _processors = {
        '.txt': TxtProcessor,
        '.pdf': PdfProcessor,
        '.docx': DocxProcessor,
        '.xlsx': XlsxProcessor,
        '.xls': XlsxProcessor,
        '.jpg': ImageProcessor,
        '.jpeg': ImageProcessor,
        '.png': ImageProcessor,
        '.bmp': ImageProcessor,
        '.tiff': ImageProcessor,
        '.gif': ImageProcessor
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._instances: Dict[str, BaseProcessor] = {}

    def get_processor(self, file_extension: str) -> Optional[BaseProcessor]:
        """根据文件扩展名获取处理器"""
        ext = file_extension.lower()

        if ext in self._instances:
            return self._instances[ext]

        if ext in self._processors:
            processor_class = self._processors[ext]
            self._instances[ext] = processor_class(self.config)
            return self._instances[ext]

        return None

    def process_file(self, file_path: str, to_markdown: bool = True) -> Dict:
        """处理单个文件，返回提取的文本和元数据"""
        ext = Path(file_path).suffix.lower()
        processor = self.get_processor(ext)

        if not processor:
            return {
                'path': file_path,
                'text': '',
                'metadata': {},
                'error': f'不支持的文件类型: {ext}'
            }

        try:
            if ext in ['.xlsx', '.xls'] and hasattr(processor, 'extract_text'):
                text = processor.extract_text(file_path, to_markdown=to_markdown)
            else:
                text = processor.extract_text(file_path)
            metadata = processor.extract_metadata(file_path)
            return {
                'path': file_path,
                'text': text,
                'metadata': metadata,
                'error': None
            }
        except Exception as e:
            return {
                'path': file_path,
                'text': '',
                'metadata': {},
                'error': str(e)
            }

    @classmethod
    def get_supported_extensions(cls) -> list:
        """获取所有支持的文件扩展名"""
        return list(cls._processors.keys())
