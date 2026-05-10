import requests
from typing import Dict, List
from .base import BaseModel


class DeepseekModel(BaseModel):
    """Deepseek接口"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get('api_key', '')
        self.model_name = config.get('model', 'deepseek-chat')
        self.base_url = config.get('base_url', 'https://api.deepseek.com/v1')
        self.timeout = config.get('timeout', 300)

    @property
    def name(self) -> str:
        return f"deepseek/{self.model_name}"

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """调用Deepseek API"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', 4096)
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Deepseek连接失败: {str(e)}")

    def check_connection(self) -> bool:
        """检查Deepseek API是否可用"""
        if not self.api_key:
            return False
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hi"}]
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
