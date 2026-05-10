from typing import Dict, List
import json
import re

from models.factory import ModelFactory


class CategoryMerger:
    """同义词合并处理器 - 将模型分类后的类别进行聚类合并"""

    MERGE_PROMPT = """你是一个高级数据分析师，专门负责对海量零散标签进行【降维聚类】。
你的核心任务是：对输入的细碎类别进行极其深度的抽象和归纳，将它们大幅压缩！

【禁止笼统分类】
不允许创造“垃圾桶”式的万金油类别。聚类后的主类名，必须是具体的业务实体或职能模块！
1. 禁用词汇库：禁止生成包含“综合”、“通用”、“业务”、“事务类”等毫无信息量的词汇。

【核心规则】
1. 强制宏观且具体的业务域归类：不要只做简单的同义词合并！必须将属于同一业务线、同一领域的词强行归为一类。
   - 示例：“财务报表”、“审计报告”、“报销单”、“税务登记” -> 合并为 “财务税务”
   - 示例："前端开发"、"UI设计"、"服务器运维" -> 合并为 "技术研发"
   - 示例：“新闻资讯”、“放假通知”、“活动公告” -> 合并为 “通知公告”
2. 目标数量控制：请务必将输入的类别列表，尽最大可能压缩到 15 到 30 个核心主类之内！
3. 主类命名规范：主类名称必须宽泛但边界清晰，字数保持在 2-8 个字之间。

【输出格式（严格按此JSON输出，不要多余字符）】
{{"merged_categories": ["宏观主类1", "宏观主类2", ...], "mapping": {{"宏观主类1": ["原类别1", "原类别2", "原类别3"]}}}}"""

    def __init__(self, model=None, config_path: str = 'config.yaml'):
        self.config_path = config_path
        if model is not None:
            self._model = model
        else:
            self.model_factory = ModelFactory(config_path)
            self._model = None

    @property
    def model(self):
        if self._model is None:
            self.model_factory = ModelFactory(self.config_path)
            self._model = self.model_factory.create_model()
        return self._model

    def merge(self, categories: List[str]) -> Dict:
        """执行同义词合并"""
        if len(categories) <= 1:
            return {
                'merged_categories': categories,
                'mapping': {cat: [cat] for cat in categories}
            }

        categories_str = '\n'.join(f'- {cat}' for cat in categories)

        prompt = f"""{self.MERGE_PROMPT}

现在请对以下列表进行深度聚类压缩：

{categories_str}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            print(f"  [Merger] Calling model for category merging...")
            response = self.model.chat(messages)
            print(f"  [Merger] Model response received, parsing...")
            result = self._parse_merge_response(response, categories)
            print(f"  [Merger] Merged from {len(categories)} to {len(result['merged_categories'])} categories")
            return result
        except Exception as e:
            print(f"  [Merger] Model call failed: {e}")
            print(f"  [Merger] Falling back to local heuristic merge...")
            return self._fallback_merge(categories)

    def _parse_merge_response(self, response: str, original_categories: List[str]) -> Dict:
        """解析合并响应"""
        clean_response = response.strip()

        json_match = re.search(r'\{[\s\S]*\}', clean_response)
        if json_match:
            clean_response = json_match.group()

        try:
            result = json.loads(clean_response)

            merged_categories = result.get('merged_categories', [])
            mapping = result.get('mapping', {})

            all_mapped = set()
            for cats in mapping.values():
                all_mapped.update(cats)

            for cat in original_categories:
                if cat not in all_mapped:
                    mapping[cat] = [cat]
                    merged_categories.append(cat)

            if len(merged_categories) == len(original_categories):
                return self._heuristic_merge(original_categories)

            return {
                'merged_categories': merged_categories,
                'mapping': mapping
            }

        except json.JSONDecodeError:
            return self._fallback_merge(original_categories)

    def _fallback_merge(self, categories: List[str]) -> Dict:
        """降级合并策略：本地做简单启发式合并"""
        return self._heuristic_merge(categories)

    def _heuristic_merge(self, categories: List[str]) -> Dict:
        """本地启发式合并 - 作为LLM的保底方案"""
        from collections import defaultdict
        import re

        mapping = defaultdict(list)
        normalized = {}

        common_suffixes = ['数据', '报告', '报表', '统计', '管理', '档案', '通知', '公告', '计划', '方案', '制度', '规定']
        domain_map = {
            '医疗': ['医疗', '卫生', '健康', '医药'],
            '财务': ['财务', '金融', '会计', '财政', '税务'],
            '人事': ['人事', '人力', '员工', '招聘'],
            '项目': ['项目', '工程', '计划'],
            '数据': ['数据', '统计', '报表', '分析'],
            '通知': ['通知', '公告', '消息', '新闻'],
            '合同': ['合同', '协议', '合作'],
            '制度': ['制度', '规定', '章程', '规范'],
            '教育': ['教育', '教学', '培训', '学习'],
            '会议': ['会议', '纪要', '记录'],
        }

        for cat in categories:
            found = False
            for domain, keywords in domain_map.items():
                for kw in keywords:
                    if kw in cat:
                        mapping[domain].append(cat)
                        normalized[cat] = domain
                        found = True
                        break
                if found:
                    break
            if not found:
                mapping[cat].append(cat)
                normalized[cat] = cat

        merged_categories = list(mapping.keys())

        return {
            'merged_categories': merged_categories,
            'mapping': {k: v for k, v in mapping.items()}
        }
