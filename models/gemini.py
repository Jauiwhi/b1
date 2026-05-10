import requests
from typing import Dict, List
from .base import BaseModel


class GeminiModel(BaseModel):
    """Google Gemini云端模型接口"""

    def __init__(self, config: Dict):
        super().__init__(config)
        api_key = config.get('api_key', '')
        self.model_name = config.get('model', 'gemini-1.5-flash')
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}"
        self.api_key = api_key
        self.timeout = config.get('timeout', 300)

    @property
    def name(self) -> str:
        return f"gemini/{self.model_name}"

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """调用Gemini API"""
        contents = self._convert_to_gemini_format(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get('temperature', 0.7),
                "maxOutputTokens": kwargs.get('max_tokens', 4096)
            }
        }

        try:
            url = f"{self.base_url}:generateContent?key={self.api_key}"
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Gemini连接失败: {str(e)}")

    def check_connection(self) -> bool:
        """检查Gemini API是否可用"""
        if not self.api_key:
            return False
        try:
            url = f"{self.base_url}:generateContent?key={self.api_key}"
            payload = {"contents": [{"parts": [{"text": "Hi"}]}]}
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _convert_to_gemini_format(self, messages: List[Dict]) -> List[Dict]:
        """转换消息为Gemini格式"""
        contents = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            gemini_role = "user" if role == "user" else "model"

            if isinstance(content, list):
                parts = []
                for part in content:
                    part_type = part.get('type')
                    if part_type == 'text':
                        parts.append({"text": part.get('text', '')})
                    elif part_type == 'image_url':
                        image_data = part.get('image_url', {}).get('url', '')
                        if image_data.startswith('data:image'):
                            parts.append({"inlineData": {
                                "mimeType": image_data.split(';')[0].replace('data:', ''),
                                "data": image_data.split(',')[1]
                            }})
                contents.append({"role": gemini_role, "parts": parts})
            else:
                contents.append({"role": gemini_role, "parts": [{"text": str(content)}]})

        return contents
