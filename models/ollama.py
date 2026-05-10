import requests
from typing import Dict, List
from .base import BaseModel


class OllamaModel(BaseModel):
    """Ollama本地模型接口"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 11434)
        self.model_name = config.get('model_name', 'llama3.2')
        self.timeout = config.get('timeout', 120)
        self.base_url = f"http://{self.host}:{self.port}"

    @property
    def name(self) -> str:
        return f"ollama/{self.model_name}"

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """调用Ollama API"""
        ollama_messages = self._convert_messages(messages)

        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": False
        }
        if 'temperature' in kwargs:
            payload["temperature"] = kwargs['temperature']
        if 'max_tokens' in kwargs:
            payload["options"] = {"num_predict": kwargs['max_tokens']}

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get('message', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Ollama连接失败: {str(e)}")

    def check_connection(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                return self.model_name in model_names
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
