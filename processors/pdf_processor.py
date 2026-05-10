from typing import Dict, List
import sys
from io import StringIO
from pathlib import Path
from .base import BaseProcessor

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


class PdfProcessor(BaseProcessor):
    """PDF文件处理器"""

    @property
    def supported_extensions(self) -> List[str]:
        return ['.pdf']

    def extract_text(self, file_path: str) -> str:
        """从PDF提取文本 - 掐头留尾"""
        if not PYPDF_AVAILABLE:
            return "[PDF处理不可用] 请安装pypdf"

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            result = self._extract_head_tail(file_path)
            return result
        except Exception as e:
            error_msg = str(e)
            if "Catalog" in error_msg or "Invalid Root" in error_msg or "Object" in error_msg:
                return f"[PDF格式错误] 该PDF文件可能损坏或格式不标准"
            return f"[PDF解析失败] {error_msg}"
        finally:
            sys.stderr = old_stderr

    def _extract_head_tail(self, file_path: str) -> str:
        """掐头留尾提取 - 只提取首尾页面"""
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        text_parts = []

        if total_pages == 0:
            return "[PDF内容为空]"

        first_page_text = self._extract_page_text(reader, 0)
        if first_page_text:
            text_parts.append(f"[首页]\n{first_page_text}")

        if total_pages > 2:
            middle_count = total_pages - 2
            text_parts.append(f"[...共{total_pages}页，已省略{middle_count}页...]")

        last_page_text = self._extract_page_text(reader, total_pages - 1)
        if last_page_text:
            text_parts.append(f"[末页]\n{last_page_text}")

        if not text_parts:
            return "[PDF内容为空]"

        return '\n\n'.join(text_parts)

    def _extract_page_text(self, reader: PdfReader, page_index: int) -> str:
        """提取指定页面的文本"""
        try:
            page = reader.pages[page_index]
            text = page.extract_text()
            return text.strip() if text else ""
        except Exception:
            return ""

    def extract_metadata(self, file_path: str) -> Dict:
        """提取PDF元数据"""
        info = self.get_file_info(file_path)
        metadata = {
            **info,
            'type': 'pdf',
            'page_count': 0,
            'title': '',
            'author': '',
            'subject': ''
        }

        if not PYPDF_AVAILABLE:
            return metadata

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            reader = PdfReader(file_path)
            metadata['page_count'] = len(reader.pages)

            if reader.metadata:
                metadata['title'] = reader.metadata.get('/Title', '')
                metadata['author'] = reader.metadata.get('/Author', '')
                metadata['subject'] = reader.metadata.get('/Subject', '')
        except Exception:
            pass
        finally:
            sys.stderr = old_stderr

        return metadata
