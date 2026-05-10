import requests
from typing import Dict, List
from .base import BaseModel


class FastChatModel(BaseModel):
    """FastChat模型接口 (OpenAI兼容格式)"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 8000)
        self.model_name = config.get('model_name', 'local-model')
        self.timeout = config.get('timeout', 120)
        self.base_url = f"http://{self.host}:{self.port}"

    @property
    def name(self) -> str:
        return f"fastchat/{self.model_name}"

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """调用FastChat API"""
        ollama_messages = self._convert_messages(messages)

        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', 4096)
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"FastChat连接失败: {str(e)}")

    def check_connection(self) -> bool:
        """检查FastChat服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """转换消息格式"""
        converted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                content = '\n'.join(text_parts)

            converted.append({"role": role, "content": content})
        return converted
