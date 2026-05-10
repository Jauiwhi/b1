import requests
from typing import Dict, List
from .base import BaseModel


class JanModel(BaseModel):
    """Jan模型接口 (OpenAI兼容格式)"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 1337)
        self.model_name = config.get('model_name', '')
        self.timeout = config.get('timeout', 120)
        self.base_url = f"http://{self.host}:{self.port}"

    @property
    def name(self) -> str:
        return f"jan/{self.model_name or 'unknown'}"

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """调用Jan API"""
        ollama_messages = self._convert_messages(messages)

        if not self.model_name:
            raise ValueError("Jan模型未配置model_name，请在config.yaml中设置")

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

            if response.status_code != 200:
                error_detail = response.text
                raise ConnectionError(f"Jan API错误 ({response.status_code}): {error_detail}")

            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Jan连接失败: {str(e)}")

    def check_connection(self) -> bool:
        """检查Jan服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('data', [])
                available_models = [m.get('id', '') for m in models]
                if available_models:
                    if not self.model_name:
                        self.model_name = available_models[0]
                    return True
            return False
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
