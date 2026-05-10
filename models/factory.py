import yaml
from typing import Dict, Optional
from .base import BaseModel
from .ollama import OllamaModel
from .lm_studio import LMStudioModel
from .vllm import VLLMModel
from .localai import LocalAIModel
from .jan import JanModel
from .fastchat import FastChatModel
from .textgen import TextGenWebUIModel
from .gemini import GeminiModel
from .chatgpt import ChatGPTModel
from .qwen import QwenModel
from .deepseek import DeepseekModel


class ModelFactory:
    """模型工厂，根据配置创建对应的模型实例"""

    _providers = {
        'ollama': OllamaModel,
        'lm_studio': LMStudioModel,
        'vllm': VLLMModel,
        'localai': LocalAIModel,
        'jan': JanModel,
        'fastchat': FastChatModel,
        'textgen': TextGenWebUIModel,
        'gemini': GeminiModel,
        'chatgpt': ChatGPTModel,
        'qwen': QwenModel,
        'deepseek': DeepseekModel
    }

    _local_providers = {'ollama', 'lm_studio', 'vllm', 'localai', 'jan', 'fastchat', 'textgen'}
    _cloud_providers = {'gemini', 'chatgpt', 'qwen', 'deepseek'}

    _default_models = {
        'ollama': 'llama3.2',
        'lm_studio': 'local-model',
        'vllm': 'local-model',
        'localai': 'local-model',
        'jan': '',
        'fastchat': 'local-model',
        'textgen': 'local-model'
    }

    _default_ports = {
        'ollama': 11434,
        'lm_studio': 1234,
        'vllm': 8000,
        'localai': 8080,
        'jan': 1337,
        'fastchat': 8000,
        'textgen': 5000
    }

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        self._model_instance: Optional[BaseModel] = None

    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        except yaml.YAMLError:
            raise ValueError(f"配置文件 {self.config_path} 格式错误")

    def create_model(self) -> BaseModel:
        """创建模型实例"""
        if self._model_instance:
            return self._model_instance

        provider = self.config.get('model_provider', 'ollama')

        if provider not in self._providers:
            raise ValueError(f"不支持的模型提供商: {provider}")

        if provider in self._local_providers:
            model_config = self.config.get('local', {}).copy()
            config_model = model_config.get('model_name', '')
            if not config_model:
                model_config['model_name'] = self._default_models.get(provider, 'default')
            if not model_config.get('port'):
                model_config['port'] = self._default_ports.get(provider, 8080)
        else:
            model_config = self.config.get('cloud', {}).get(provider, {}).copy()
            # 确保云端模型也能获取到 timeout 配置
            model_config['timeout'] = self.config.get('local', {}).get('timeout', 300)

        model_class = self._providers[provider]
        self._model_instance = model_class(model_config)

        return self._model_instance

    def check_model_connection(self) -> Dict:
        """检查模型连接状态"""
        model = self.create_model()
        is_connected = model.check_connection()
        supports_vision = False

        if is_connected:
            supports_vision = model.supports_vision

        result = {
            'provider': self.config.get('model_provider'),
            'model_name': model.name,
            'connected': is_connected,
            'supports_vision': supports_vision
        }

        if not is_connected:
            result['error'] = self._get_connection_error_hint(provider)

        return result

    def _get_connection_error_hint(self, provider: str) -> str:
        """获取连接错误提示"""
        hints = {
            'ollama': "请确保Ollama服务正在运行，执行: ollama serve",
            'lm_studio': "请确保LM Studio正在运行并加载模型",
            'vllm': "请确保vLLM服务正在运行",
            'localai': "请确保LocalAI服务正在运行",
            'jan': "请确保Jan正在运行，并在config.yaml中设置正确的model_name",
            'fastchat': "请确保FastChat服务正在运行",
            'textgen': "请确保Text Generation WebUI正在运行",
            'gemini': "请检查config.yaml中的Gemini API Key是否正确",
            'chatgpt': "请检查config.yaml中的OpenAI API Key是否正确",
            'qwen': "请检查config.yaml中的阿里云API Key是否正确",
            'deepseek': "请检查config.yaml中的Deepseek API Key是否正确"
        }
        return hints.get(provider, "请检查模型配置")

    @classmethod
    def get_available_providers(cls) -> Dict:
        """获取可用的模型提供商列表"""
        return {
            'local': list(cls._local_providers),
            'cloud': list(cls._cloud_providers)
        }
