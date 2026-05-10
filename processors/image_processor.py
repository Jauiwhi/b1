from typing import Dict, List
import base64
from pathlib import Path
from PIL import Image
from .base import BaseProcessor

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class ImageProcessor(BaseProcessor):
    """图片处理器，支持OCR识别"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.ocr_language = config.get('ocr_language', 'chi_sim+eng') if config else 'chi_sim+eng'

    @property
    def supported_extensions(self) -> List[str]:
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

    def extract_text(self, file_path: str, use_ocr: bool = True) -> str:
        """从图片提取文本，仅识别顶部标题区域"""
        if not use_ocr:
            return self._extract_image_description(file_path)

        if not TESSERACT_AVAILABLE:
            return "[OCR不可用] 请安装tesseract-ocr和pytesseract"

        try:
            return self._extract_with_tesseract(file_path)
        except Exception as e:
            return f"[OCR识别失败: {str(e)}]"

    def _extract_with_tesseract(self, file_path: str) -> str:
        """使用Tesseract提取文本，仅识别顶部标题"""
        image = Image.open(file_path)
        width, height = image.size

        top_height = min(height // 5, 100)
        top_region = image.crop((0, 0, width, top_height))

        top_text = pytesseract.image_to_string(top_region, lang=self.ocr_language)
        top_text = top_text.strip() if top_text.strip() else "[未识别到顶部标题]"

        return f"【顶部标题】\n{top_text}"

    def _extract_image_description(self, file_path: str) -> str:
        """提取图片描述（用于视觉模型）"""
        try:
            with open(file_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            path = Path(file_path)
            ext = path.suffix.lower()[1:]
            mime_type = f'image/{ext}' if ext != 'jpg' else 'image/jpeg'
            return f"[图片数据: data:{mime_type};base64,...]"
        except Exception as e:
            return f"[图片读取失败: {str(e)}]"

    def get_base64_image(self, file_path: str) -> str:
        """获取图片的base64编码"""
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def extract_metadata(self, file_path: str) -> Dict:
        """提取图片元数据"""
        info = self.get_file_info(file_path)
        metadata = {
            **info,
            'type': 'image',
            'width': 0,
            'height': 0,
            'mode': ''
        }

        try:
            with Image.open(file_path) as img:
                metadata['width'] = img.width
                metadata['height'] = img.height
                metadata['mode'] = img.mode
        except Exception:
            pass

        return metadata
