from typing import Dict, List
from pathlib import Path
import json
import shutil
from collections import defaultdict

from classifier.batch_classifier import BatchClassifier
from processors.factory import ProcessorFactory


class Problem2Solver:
    """问题二：数据集2/3处理和分类"""

    CLASSIFICATION_PROMPT = """你是一个专业的办公文档分类专家。根据已建立的分类体系，对新文件进行分类。

【核心分类原则】
1. 标题优先级最高：很多文件由"顶部标题 + 大段正文/大片表格"组成（如"XX地区旅游业收入"下面全是表格）。请务必优先识别并依据文件最上方的"标题"来界定业务属性，不要被底下繁杂的数据干扰。
2. 只能使用以下已知类别，禁止自创类别！
已知类别列表：
{categories}
3. 如果文件确实无法归入任何已知类别，标记为"未分类"
4. 绝对禁止输出"图片素材"、"扫描件"、"图片元数据"等无意义类别

【输入与输出规范】
我会以"文件序号：文件内容"的形式提供数据。
请严格输出一个标准的 JSON 对象，Key 必须是输入时对应的"文件序号"（纯数字格式），Value 是你判断的"分类名称"。不要输出任何额外的解释性文字！

期待的 JSON 输出格式示例：
{{"1": "人事档案", "2": "财务报告", "3": "未分类"}}

以下是本次需要分类的文件内容："""

    DEFAULT_CATEGORIES_PATH = 'output/问题一结果/分类结果.json'

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.classifier = BatchClassifier(config_path)
        self.processor_factory = ProcessorFactory()
        self._known_categories: List[str] = []
        self._dataset3_files: List[Dict] = []
        self._script_dir = Path(__file__).parent.parent

    def check_model(self) -> Dict:
        """检查模型连接"""
        return self.classifier.check_connection()

    def set_known_categories(self, categories: List[str]):
        """设置已知的分类类别"""
        self._known_categories = categories

    def _load_categories_from_problem1(self, categories_path: str = None) -> List[str]:
        """从problem1的结果JSON中加载类别"""
        if categories_path is None:
            categories_path = self.DEFAULT_CATEGORIES_PATH

        path = Path(categories_path)
        if not path.exists():
            path = self._script_dir / categories_path

        if not path.exists():
            print(f"Warning: Problem1 result not found at {categories_path}")
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            categories = data.get('categories', [])
            print(f"Loaded {len(categories)} categories from Problem1 result")

            category_merge = data.get('category_merge', {})
            if category_merge:
                merged = category_merge.get('merged_categories', [])
                if merged:
                    categories = merged
                    print(f"Using merged categories: {len(categories)}")

            return categories
        except Exception as e:
            print(f"Failed to load categories: {e}")
            return []

    def _find_input_path(self, dataset_name: str, is_file: bool = False) -> str:
        """查找数据集路径，使用关键词模糊匹配"""
        search_keyword = dataset_name.split('：')[0]

        for base_dir in [self._script_dir / 'input', self._script_dir]:
            if not base_dir.exists():
                continue
            for item in base_dir.iterdir():
                if item.name.startswith('.~lock.'):
                    continue
                if is_file and item.is_file() and search_keyword in item.name:
                    return str(item)
                elif not is_file and item.is_dir() and search_keyword in item.name:
                    return str(item)

        if is_file:
            return str(self._script_dir / 'input' / (search_keyword + '.xlsx'))
        return str(self._script_dir / 'input' / dataset_name)

    def solve(self, dataset2_path: str, dataset3_path: str,
              output_dir: str = 'output/问题二结果', image_mode: str = 'auto',
              problem1_result_path: str = None) -> Dict:
        """执行问题二求解"""
        print("=" * 60)
        print("Problem 2: Dataset2/3 Classification")
        print("=" * 60)
        print(f"Image mode: {image_mode}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        connection_info = self.check_model()
        print(f"Model: {connection_info['model_name']}")
        print(f"Vision Support: {'Yes' if connection_info.get('supports_vision') else 'No'}")
        print()

        if not connection_info['connected']:
            return {'success': False, 'error': 'Model connection failed'}

        self.classifier.set_image_mode(image_mode)

        if not self._known_categories:
            self._known_categories = self._load_categories_from_problem1(problem1_result_path)

        if not self._known_categories:
            print("Warning: No known categories found!")
            self._known_categories = ["未分类"]

        if not dataset2_path:
            dataset2_path = self._find_input_path("数据集2")
        if not dataset3_path:
            dataset3_path = self._find_input_path("数据集3", is_file=True)

        print(f"Using {len(self._known_categories)} categories for classification")
        print()

        print("Processing Dataset2...")
        dataset2_results = self._process_and_classify_dataset2(dataset2_path)
        print()

        print("Processing Dataset3 (xlsx -> txt)...")
        dataset3_results = self._process_dataset3(dataset3_path, output_dir)
        print(f"Dataset3 processed: {len(dataset3_results)} files")

        if dataset3_results:
            dataset3_classified = self._classify_dataset3(dataset3_results)
        else:
            dataset3_classified = []

        all_classified = dataset2_results + dataset3_classified
        all_results = {}

        for item in dataset2_results:
            all_results[item['file_path']] = {
                'category': item['category'],
                'extension': item['extension'],
                'dataset': 'dataset2'
            }

        for item in dataset3_classified:
            all_results[item['file_path']] = {
                'category': item['category'],
                'extension': item.get('extension', '.txt'),
                'dataset': 'dataset3'
            }

        category_distribution = self._calculate_distribution(all_results)
        print()
        print(f"Classification complete! Total {len(all_results)} files in {len(category_distribution)} categories:")
        for cat, count in sorted(category_distribution.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

        self._organize_files(all_results, output_dir)

        result_file = Path(output_dir) / '分类结果.json'
        result_file.write_text(json.dumps({
            'categories': list(self._known_categories),
            'results': all_results,
            'statistics': category_distribution,
            'dataset2_count': len(dataset2_results),
            'dataset3_count': len(dataset3_classified)
        }, ensure_ascii=False, indent=2), encoding='utf-8')

        summary = {
            'success': True,
            'total_files': len(all_results),
            'dataset2_count': len(dataset2_results),
            'dataset3_count': len(dataset3_classified),
            'categories': self._known_categories,
            'category_distribution': category_distribution,
            'output_dir': output_dir,
            'result_file': str(result_file)
        }

        print()
        print("=" * 60)
        print("Problem 2 Complete")
        print(f"Results: {result_file}")
        print("=" * 60)

        return summary

    def _process_and_classify_dataset2(self, dataset2_path: str):
        """处理数据集2：按文件类型分批，用32k token限制批量分类"""
        if not Path(dataset2_path).exists():
            print(f"Dataset2 directory not found: {dataset2_path}")
            return []

        files_by_ext = self.classifier.collect_files(dataset2_path)
        print(f"Found file types:")
        for ext, files in sorted(files_by_ext.items()):
            print(f"  {ext}: {len(files)} files")
        print()

        all_results = []
        extensions = ['.txt', '.pdf', '.docx', '.jpg', '.jpeg', '.png', '.xlsx']

        for ext in extensions:
            if ext not in files_by_ext:
                continue

            file_paths = files_by_ext[ext]
            print(f"Processing {ext} files ({len(file_paths)} total)...")

            processed_files = []
            for path in file_paths:
                file_data = self.classifier.process_file(path)
                if file_data.get('text') or file_data.get('image_base64'):
                    processed_files.append(file_data)

            if not processed_files:
                print(f"  No valid files")
                continue

            categories_str = '\n'.join(f'- {cat}' for cat in self._known_categories)
            prompt = self.CLASSIFICATION_PROMPT.format(categories=categories_str)

            results = self.classifier.classify_batch(processed_files, prompt)

            for result in results:
                category = result.get('category', '未分类')
                file_path = result.get('file_path', '')
                all_results.append({
                    'file_path': file_path,
                    'category': category,
                    'extension': ext
                })

            print(f"  Done - {len(results)} files classified")

        return all_results

    def _process_dataset3(self, xlsx_path: str, output_dir: str) -> List[Dict]:
        """处理数据集3：xlsx转换为txt"""
        if not Path(xlsx_path).exists():
            print(f"Dataset3 file not found: {xlsx_path}")
            return []

        output_subdir = str(Path(output_dir) / '数据集3文件夹')
        Path(output_subdir).mkdir(parents=True, exist_ok=True)

        xlsx_processor = self.processor_factory.get_processor('.xlsx')
        if not xlsx_processor:
            print("XLSX processor not available")
            return []

        try:
            rows = xlsx_processor.extract_rows(xlsx_path)
        except Exception as e:
            print(f"Failed to read xlsx: {e}")
            return []

        results = []
        for row_idx, row in enumerate(rows, 1):
            file_id = None
            has_content = False

            for id_key in ['编号', '文件编号', 'id', 'ID']:
                if id_key in row and row[id_key]:
                    file_id = row[id_key]
                    break

            for key, value in row.items():
                if key.startswith('_') or value is None:
                    continue
                if value != '' and str(value).strip():
                    has_content = True
                    break

            if not has_content or not file_id:
                continue

            file_name = f"数据集3_{file_id}"

            content_parts = []
            for key, value in row.items():
                if key.startswith('_') or value is None:
                    continue
                content_parts.append(f"{key}: {value}")

            content = '\n'.join(content_parts)

            txt_path = str(Path(output_subdir) / f"{file_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self._dataset3_files.append({
                'file_name': file_name,
                'file_path': txt_path,
                'content': content,
                'extension': '.txt'
            })
            results.append(self._dataset3_files[-1])

        print(f"数据集3处理完成，共生成 {len(results)} 个txt文件，保存在: {output_subdir}")
        return results

    def _classify_dataset3(self, dataset3_data: List[Dict]) -> List[Dict]:
        """对数据集3进行分类"""
        if not dataset3_data:
            return []

        categories_str = '\n'.join(f'- {cat}' for cat in self._known_categories)
        prompt = self.CLASSIFICATION_PROMPT.format(categories=categories_str)

        processed = []
        for item in dataset3_data:
            file_data = {
                'path': item['file_path'],
                'text': item.get('content', ''),
                'image_base64': None,
                'metadata': {}
            }
            processed.append(file_data)

        results = self.classifier.classify_batch(processed, prompt)

        classified = []
        for item, result in zip(dataset3_data, results):
            item['category'] = result.get('category', '未分类')
            classified.append(item)

        return classified

    def _organize_files(self, results: Dict, output_dir: str):
        """根据分类结果整理文件到分类文件夹"""
        categories = set()
        for info in results.values():
            cat = info.get('category', '')
            if cat and not cat.startswith('ERROR'):
                categories.add(cat)

        base_path = Path(output_dir)

        for category in categories:
            (base_path / '分类文件夹' / category).mkdir(parents=True, exist_ok=True)

        organized_count = 0
        for file_path, info in results.items():
            category = info.get('category', '未分类')

            if category.startswith('ERROR') or category == '未分类':
                category = '未分类'
                dest_dir = base_path / '分类文件夹' / '未分类'
            else:
                dest_dir = base_path / '分类文件夹' / category

            dest_dir.mkdir(parents=True, exist_ok=True)

            src_path = Path(file_path)
            dest_path = dest_dir / src_path.name

            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
                counter += 1

            try:
                shutil.copy2(file_path, dest_path)
                organized_count += 1
            except Exception as e:
                print(f"Copy failed: {file_path}: {e}")

        print()
        print(f"Organized {organized_count} files into category folders")

    def _calculate_distribution(self, results: Dict) -> Dict[str, int]:
        """计算分类分布"""
        distribution = defaultdict(int)
        for info in results.values():
            category = info.get('category', '未分类')
            distribution[category] += 1
        return dict(distribution)
