from typing import Dict, List
from pathlib import Path
import json
import shutil
from collections import defaultdict

from classifier.batch_classifier import BatchClassifier
from classifier.merger import CategoryMerger


class Problem1Solver:
    """问题一：数据集1分类求解器"""

    CATEGORY_PROMPT = """你是一个专业的办公文档分类专家。我将为你提供一批文件的内容（可能是纯文本，也可能是图片识别后的内容）。请仔细分析并将其归类到最合适的业务类别。

【核心分类原则】
1. 标题优先级最高：很多文件由"顶部标题 + 大段正文/大片表格"组成（如"XX地区旅游业收入"下面全是表格）。请务必优先识别并依据文件最上方的"标题"来界定业务属性，不要被底下繁杂的数据干扰。
2. 内容优先级高（仅次于标题）：无论文件原来是Word、Excel还是图片，请一律根据其表达的"实际业务内容"分类。绝对禁止输出"图片素材"、"扫描件"、"图片元数据"等无意义类别。
3. 动态命名：类别名称由你根据内容精准提炼，长度限制在 2-8 个字以内（如：人事档案、财务报告、会议记录、数据报表、项目计划等）。
4. 特殊兜底机制：
   - 纯 URL 链接列表（无其他实质内容） -> 归为"链接索引"
   - 内容极度模糊 或 完全无法判断业务属性 -> 归为"未分类"

【正确分类示例】
- 文件包含标题"XX公司人员情况表"及其下方表格 -> 人事档案
- 文件包含标题"XX年财务收支表"及其下方文字 -> 财务报告
- 文件包含标题"按行业分城镇就业人员数"及其下方表格 -> 人力资源统计
- 文件内容为项目进度甘特图 -> 项目计划

【输入与输出规范】
我会以"文件序号：文件内容"的形式提供数据。
请严格输出一个标准的 JSON 对象，Key 必须是输入时对应的"文件序号"（纯数字格式），Value 是你判断的"分类名称"。不要输出任何额外的解释性文字！

期待的 JSON 输出格式示例：
{
  "1": "人事档案",
  "2": "财务报告",
  "3": "未分类"
}

以下是本次需要分类的文件内容：
"""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.classifier = BatchClassifier(config_path)
        self._categories: List[str] = []
        self._category_descriptions: Dict[str, str] = {}

    def check_model(self) -> Dict:
        """检查模型连接和视觉支持"""
        return self.classifier.check_connection()

    def solve(self, dataset1_path: str, output_dir: str = 'output/问题一结果', image_mode: str = 'auto') -> Dict:
        """执行问题一求解"""
        print("=" * 60)
        print("Problem 1: Dataset1 Classification")
        print("=" * 60)
        print(f"Image mode: {image_mode}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        connection_info = self.check_model()
        print(f"Model: {connection_info['model_name']}")
        print(f"Vision Support: {'Yes' if connection_info.get('supports_vision') else 'No'}")
        print()

        if not connection_info['connected']:
            return {'success': False, 'error': connection_info.get('error', 'Model connection failed')}

        self.classifier.set_image_mode(image_mode)

        files_by_ext = self.classifier.collect_files(dataset1_path)
        print(f"Found file types:")
        for ext, files in sorted(files_by_ext.items()):
            print(f"  {ext}: {len(files)} files")
        print()

        all_results = {}
        category_count = defaultdict(int)

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

            results = self._batch_classify(processed_files, self.CATEGORY_PROMPT)

            for result in results:
                category = result['category']
                file_path = result['file_path']
                all_results[file_path] = {
                    'category': category,
                    'extension': ext
                }
                category_count[category] += 1

            print(f"  Done - {len(results)} files classified")

        self._categories = list(category_count.keys())

        print()
        print("Merging similar categories...")
        categories_for_merge = [c for c in self._categories if c != "未分类"]
        print(f"  Filtered out '未分类', merging {len(categories_for_merge)} categories...")
        merge_result = self._merge_categories(categories_for_merge)
        merge_mapping = merge_result['mapping']
        new_categories = merge_result['merged_categories']
        if "未分类" not in new_categories and "未分类" in self._categories:
            new_categories.append("未分类")

        print(f"  Original categories: {len(self._categories)}")
        print(f"  Merged categories: {len(new_categories)}")
        for new_cat, originals in merge_mapping.items():
            if len(originals) > 1:
                print(f"    {new_cat} = {originals}")

        all_results = self._remap_results(all_results, merge_mapping)
        category_count = self._count_categories(all_results)

        print()
        print(f"Classification complete! {len(new_categories)} categories found:")
        for cat, count in sorted(category_count.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count} files")

        self._categories = new_categories
        self._organize_files(all_results, output_dir)

        result_file = Path(output_dir) / '分类结果.json'
        result_file.write_text(json.dumps({
            'categories': list(self._categories),
            'category_descriptions': self._category_descriptions,
            'results': all_results,
            'statistics': dict(category_count),
            'category_merge': {
                'merged_categories': new_categories,
                'mapping': merge_mapping
            }
        }, ensure_ascii=False, indent=2), encoding='utf-8')

        summary = {
            'success': True,
            'total_files': len(all_results),
            'total_categories': len(self._categories),
            'category_distribution': dict(category_count),
            'output_dir': output_dir,
            'result_file': str(result_file),
            'categories': list(self._categories)
        }

        print()
        print("=" * 60)
        print("Problem 1 Complete")
        print(f"Results: {result_file}")
        print("=" * 60)

        return summary

    def _batch_classify(self, files: List[Dict], prompt: str) -> List[Dict]:
        """批量分类 - 使用classifier的方法"""
        return self.classifier.classify_batch(files, prompt)

    def _organize_files(self, results: Dict, output_dir: str):
        """根据分类结果整理文件"""
        categories = set(r['category'] for r in results.values() if not r['category'].startswith('处理失败'))
        base_path = Path(output_dir)

        for category in categories:
            (base_path / '分类文件夹' / category).mkdir(parents=True, exist_ok=True)

        for file_path, info in results.items():
            category = info['category']

            if category.startswith('ERROR') or category == '未分类' or category == '处理失败':
                category = '未分类'

            src_path = Path(file_path)
            dest_dir = base_path / '分类文件夹' / category
            dest_path = dest_dir / src_path.name

            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
                counter += 1

            try:
                shutil.copy2(file_path, dest_path)
            except Exception as e:
                print(f"Copy failed: {file_path}: {e}")

    def get_categories(self) -> List[str]:
        """获取识别的类别列表"""
        return self._categories

    def get_category_descriptions(self) -> Dict[str, str]:
        """获取类别描述"""
        return self._category_descriptions

    def _merge_categories(self, categories: List[str]) -> Dict:
        """调用LLM合并同义词"""
        merger = CategoryMerger(model=self.classifier.model, config_path=self.config_path)
        return merger.merge(categories)

    def _remap_results(self, results: Dict, mapping: Dict) -> Dict:
        """根据映射关系更新分类结果"""
        for file_path, info in results.items():
            original_category = info['category']
            for merged_cat, original_cats in mapping.items():
                if original_category in original_cats:
                    info['category'] = merged_cat
                    break
        return results

    def _count_categories(self, results: Dict) -> Dict[str, int]:
        """统计各类别文件数量"""
        counts = defaultdict(int)
        for info in results.values():
            cat = info.get('category', '未分类')
            counts[cat] += 1
        return dict(counts)
