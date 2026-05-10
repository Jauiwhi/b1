from typing import Dict, List
import json
import base64
import re
from pathlib import Path
from collections import defaultdict

from models.factory import ModelFactory
from processors.factory import ProcessorFactory


class BatchClassifier:
    """批量分类器，支持按文件类型分批处理，32k上下文限制"""

    def __init__(self, config_path: str = 'config.yaml'):
        self.model_factory = ModelFactory(config_path)
        self.config = self.model_factory.config
        self.processor_factory = ProcessorFactory(self.config.get('processing', {}))

        self.max_context_tokens = self.config.get('processing', {}).get('max_context_tokens', 32000)
        self.batch_size = self.config.get('processing', {}).get('batch_size', 50)
        self.vision_config = self.config.get('vision', {})
        self.image_mode = 'ocr'

        self._model = None

    def set_image_mode(self, mode: str):
        """设置图片识别模式: vision=AI视觉, ocr=OCR+AI, skip=跳过图片"""
        self.image_mode = mode

    @property
    def model(self):
        """获取模型实例"""
        if self._model is None:
            self._model = self.model_factory.create_model()
        return self._model

    def check_connection(self) -> Dict:
        """检查模型连接"""
        return self.model_factory.check_model_connection()

    def collect_files(self, directory: str, extensions: List[str] = None) -> Dict[str, List[str]]:
        """收集指定目录下的文件，按扩展名分组"""
        files_by_type = defaultdict(list)

        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if extensions is None or ext in extensions:
                    files_by_type[ext].append(str(file_path))

        return dict(files_by_type)

    def process_file(self, file_path: str) -> Dict:
        """处理单个文件"""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if self.image_mode == 'skip':
                return {
                    'path': file_path,
                    'text': '[已跳过图片]',
                    'success': True,
                    'skipped': True
                }
            return self._process_image_file(file_path)
        else:
            return self._process_text_file(file_path)

    def _process_text_file(self, file_path: str) -> Dict:
        """处理文本类文件"""
        result = self.processor_factory.process_file(file_path)
        text = result.get('text', '')
        text = self._truncate_for_context(text, max_chars=800)
        return {
            'path': file_path,
            'text': text,
            'metadata': result.get('metadata', {}),
            'success': result.get('error') is None,
            'error': result.get('error')
        }

    def _process_image_file(self, file_path: str) -> Dict:
        """处理图片文件"""
        try:
            path = Path(file_path)
            ext = path.suffix.lower()[1:]
            mime_type = f'image/{ext}' if ext != 'jpg' else 'image/jpeg'

            result = self.processor_factory.process_file(file_path)
            ocr_text = result.get('text', '')
            ocr_text = self._truncate_for_context(ocr_text, max_chars=800)

            use_vision = self.image_mode == 'vision'

            if use_vision:
                with open(file_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                return {
                    'path': file_path,
                    'text': ocr_text,
                    'image_base64': img_data,
                    'mime_type': mime_type,
                    'metadata': result.get('metadata', {}),
                    'success': result.get('error') is None,
                    'error': result.get('error')
                }
            else:
                return {
                    'path': file_path,
                    'text': ocr_text or '[无法识别的图片]',
                    'image_base64': None,
                    'mime_type': mime_type,
                    'metadata': result.get('metadata', {}),
                    'success': result.get('error') is None,
                    'error': result.get('error')
                }
        except Exception as e:
            return {
                'path': file_path,
                'text': '',
                'image_base64': None,
                'metadata': {},
                'success': False,
                'error': str(e)
            }

    def _estimate_file_tokens(self, file_data: Dict) -> int:
        """估算文件的token数量 - 截断后按800字符估算"""
        if file_data.get('image_base64'):
            return 300

        text = file_data.get('text', '')
        char_count = min(len(text), 800)
        return int(char_count / 1.5)

    def _truncate_for_context(self, text: str, max_chars: int = 800) -> str:
        """掐头去尾保留法 - 保留首部和尾部关键信息"""
        if len(text) <= max_chars:
            return text

        head_chars = int(max_chars * 0.6)
        tail_chars = int(max_chars * 0.3)

        head = text[:head_chars]
        tail = text[-tail_chars:] if tail_chars > 0 else ""

        return f"{head}\n...\n{tail}"

    def _pack_files_to_context(self, files: List[Dict]) -> List[List[Dict]]:
        """将文件打包到上下文限制内"""
        batches = []
        current_batch = []
        current_tokens = 0

        for file_data in files:
            if file_data.get('skipped'):
                continue

            file_tokens = self._estimate_file_tokens(file_data)

            if current_tokens + file_tokens > self.max_context_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(file_data)
            current_tokens += file_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def _classify_batch_json(self, files: List[Dict], category_prompt: str) -> List[Dict]:
        """批量分类 - JSON结构化输出"""
        batches = self._pack_files_to_context(files)

        print(f"    Packed into {len(batches)} context batches")

        results = []
        for batch_idx, batch_files in enumerate(batches, 1):
            print(f"    Processing batch {batch_idx}/{len(batches)} ({len(batch_files)} files)...")

            batch_results = self._classify_context_batch(batch_files, category_prompt)
            results.extend(batch_results)

        return results

    def _classify_context_batch(self, files: List[Dict], category_prompt: str) -> List[Dict]:
        """对一个上下文批次进行分类，强制JSON输出"""
        prompt = self._build_batch_prompt_json(files, category_prompt)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.model.chat(messages)
            return self._parse_json_response(response, files)
        except Exception as e:
            return [{"file_path": f['path'], "category": "处理失败"} for f in files]

    def _build_batch_prompt_json(self, files: List[Dict], category_prompt: str) -> str:
        """构建批量分类提示 - 要求JSON结构化输出"""
        file_contents = []
        for idx, file_data in enumerate(files, 1):
            file_path = Path(file_data['path'])
            file_name = file_path.name
            text = file_data.get('text', '') or "[空内容]"

            truncated = self._truncate_for_context(text, max_chars=800)
            file_contents.append(f"文件{idx}: {file_name}\n内容: {truncated}")

        combined = "\n\n".join(file_contents)

        prompt = f"""{category_prompt}

请分析以下所有文件的内容，分别为每个文件分类。

{combined}

请严格输出JSON格式，键为文件编号字符串，值为分类结果。
示例格式：
{{"1": "财务报表", "2": "人事通知", "3": "合同协议"}}

请只输出JSON，不要其他内容："""

        return prompt

    def _parse_json_response(self, response: str, files: List[Dict]) -> List[Dict]:
        """解析JSON格式的分类响应"""
        clean_response = response.strip()

        json_match = re.search(r'\{[\s\S]*\}', clean_response)
        if json_match:
            clean_response = json_match.group()

        try:
            result_map = json.loads(clean_response)

            results = []
            for idx in range(1, len(files) + 1):
                file_path = files[idx - 1]['path']
                category = result_map.get(str(idx), "未分类")

                if not category or len(category) > 15:
                    category = "未分类"

                results.append({"file_path": file_path, "category": category})

            return results

        except json.JSONDecodeError:
            clean_text = re.sub(r'^```json\s*', '', clean_response)
            clean_text = re.sub(r'^```\s*', '', clean_text)
            clean_text = re.sub(r'\s*```$', '', clean_text)

            try:
                result_map = json.loads(clean_text)

                results = []
                for idx in range(1, len(files) + 1):
                    file_path = files[idx - 1]['path']
                    category = result_map.get(str(idx), "未分类")

                    if not category or len(category) > 15:
                        category = "未分类"

                    results.append({"file_path": file_path, "category": category})

                return results

            except json.JSONDecodeError:
                return [{"file_path": f['path'], "category": "未分类"} for f in files]

    def classify_by_type(self, directory: str, category_prompt: str,
                        extensions: List[str] = None) -> Dict[str, List[Dict]]:
        """按文件类型分批分类"""
        files_by_type = self.collect_files(directory, extensions)
        all_results = {}

        for ext, file_paths in sorted(files_by_type.items()):
            print(f"  Processing {ext} files ({len(file_paths)} total)...")
            processed_files = []

            for file_path in file_paths:
                file_data = self.process_file(file_path)
                if file_data.get('skipped'):
                    all_results.setdefault(ext, []).append({"file_path": file_path, "category": "已跳过"})
                    continue
                if file_data.get('text') or file_data.get('image_base64'):
                    processed_files.append(file_data)

            if not processed_files:
                continue

            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                if self.image_mode == 'vision':
                    print(f"    Using AI vision...")
                elif self.image_mode == 'ocr':
                    print(f"    Using OCR...")
                else:
                    print(f"    Skipping images...")

            results = self._classify_batch_json(processed_files, category_prompt)
            all_results[ext] = results

        return all_results

    def classify_batch(self, files: List[Dict], category_prompt: str) -> List[Dict]:
        """批量分类方法 - 返回List[Dict]"""
        return self._classify_batch_json(files, category_prompt)

    def save_results(self, results: Dict, output_path: str):
        """保存分类结果"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    def create_category_folders(self, results: Dict, base_dir: str):
        """根据分类结果创建文件夹并整理文件"""
        created_folders = set()
        base_path = Path(base_dir)

        for ext, file_results in results.items():
            for result in file_results:
                file_path = result.get('file_path', '')
                category = result.get('category', '未分类')

                if category.startswith('ERROR') or category in ['未分类', '已跳过', '处理失败']:
                    category = '未分类'
                if category.startswith('处理失败'):
                    category = '处理失败'

                category_folder = base_path / category
                if str(category_folder) not in created_folders:
                    category_folder.mkdir(parents=True, exist_ok=True)
                    created_folders.add(str(category_folder))

                src_path = Path(file_path)
                dest_path = category_folder / src_path.name

                counter = 1
                while dest_path.exists():
                    dest_path = category_folder / f"{src_path.stem}_{counter}{src_path.suffix}"
                    counter += 1

                try:
                    import shutil
                    shutil.copy2(file_path, dest_path)
                except Exception as e:
                    print(f"Copy failed: {file_path} -> {dest_path}: {e}")
