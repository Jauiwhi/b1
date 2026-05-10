from typing import Dict, List, Tuple
from pathlib import Path
import json
from collections import defaultdict
import re


class EvaluationMetrics:
    """评价指标计算器 - 基于规则实现"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def _calculate_category_purity(self, predictions: List[Dict], target_category: str) -> float:
        """计算类别纯度 - 基于类别内关键词重叠度"""
        category_files = [p for p in predictions if p.get('category') == target_category]
        if len(category_files) <= 1:
            return 1.0
        
        # 统计该类别中的关键词频率
        all_words = []
        for f in category_files:
            content = f.get('content', '')
            if content:
                words = content.split()
                all_words.extend([w for w in words if len(w) >= 2])
        
        if not all_words:
            return 0.5
        
        # 计算词频
        word_freq = defaultdict(int)
        for w in all_words:
            word_freq[w] += 1
        
        # 计算一致性：高频词占比越高，纯度越高
        total_words = len(all_words)
        high_freq_words = sum(1 for w, c in word_freq.items() if c >= len(category_files) * 0.3)
        purity = high_freq_words / len(word_freq) if word_freq else 0
        
        return purity

    def calculate_comprehensive_metrics(self, predictions: List[Dict]) -> Dict:
        """计算综合评价指标 - 适用于无监督分类"""
        if not predictions:
            return {
                'accuracy': 0,
                'purity': 0,
                'silhouette_score': 0,
                'interpretability': 0,
                'transferability': 0,
                'comprehensive_score': 0,
                'category_metrics': {},
                'confusion_summary': {}
            }

        total = len(predictions)
        valid_count = sum(1 for p in predictions if p.get('category') and 
                         '未分类' not in p.get('category', '') and 
                         not p.get('category', '').startswith('ERROR'))

        accuracy = valid_count / total if total > 0 else 0

        categories = set(p.get('category', '') for p in predictions if p.get('category'))
        category_counts = defaultdict(int)
        for p in predictions:
            cat = p.get('category', '')
            if cat:
                category_counts[cat] += 1

        category_metrics = {}
        purities = []
        
        for cat in categories:
            if cat == '未分类' or cat.startswith('ERROR'):
                continue
            
            count = category_counts[cat]
            ratio = count / total
            
            purity = self._calculate_category_purity(predictions, cat)
            category_metrics[cat] = {
                'count': count,
                'ratio': ratio,
                'purity': round(purity, 4)
            }
            purities.append(purity)

        avg_purity = sum(purities) / len(purities) if purities else 0

        interpretability = self._calculate_interpretability(predictions)
        transferability = self._calculate_transferability(predictions)
        silhouette_score = self._calculate_silhouette_score(predictions)

        comprehensive_score = (
            self.config.get('accuracy_weight', 0.4) * accuracy +
            self.config.get('interpretability_weight', 0.3) * interpretability +
            self.config.get('transferability_weight', 0.3) * transferability
        )

        unclassified_count = sum(1 for p in predictions if '未分类' in p.get('category', ''))
        error_count = sum(1 for p in predictions if p.get('category', '').startswith('ERROR'))
        multi_category_count = sum(1 for p in predictions if self._has_multi_category_features(p))

        confusion_summary = {
            'total': total,
            'valid_classified': valid_count,
            'unclassified': unclassified_count,
            'error': error_count,
            'multi_category_features': multi_category_count,
            'valid_ratio': valid_count / total if total > 0 else 0,
            'unclassified_ratio': unclassified_count / total if total > 0 else 0
        }

        return {
            'accuracy': round(accuracy, 4),
            'purity': round(avg_purity, 4),
            'silhouette_score': round(silhouette_score, 4),
            'interpretability': round(interpretability, 4),
            'transferability': round(transferability, 4),
            'comprehensive_score': round(comprehensive_score, 4),
            'category_metrics': category_metrics,
            'confusion_summary': confusion_summary
        }

    def _has_multi_category_features(self, prediction: Dict) -> bool:
        """检查文件是否具有多类别特征"""
        content = prediction.get('content', '')
        if not content:
            return False
        
        category_indicators = {
            '人事': ['人事', '员工', '招聘', '培训', '考核', '薪资'],
            '财务': ['财务', '资金', '预算', '报销', '审计', '账目'],
            '技术': ['技术', '研发', '开发', '系统', '软件', '硬件'],
            '行政': ['行政', '办公', '后勤', '物业', '设备'],
            '法务': ['法律', '合同', '协议', '诉讼', '合规']
        }
        
        matched = 0
        for cat_name, keywords in category_indicators.items():
            if any(kw in content for kw in keywords):
                matched += 1
        
        return matched >= 2

    def _calculate_interpretability(self, predictions: List[Dict]) -> float:
        """计算可解释性得分 - 有效分类的比例"""
        if not predictions:
            return 0.0

        unclassified_count = sum(1 for p in predictions if '未分类' in p.get('category', ''))
        error_count = sum(1 for p in predictions if p.get('category', '').startswith('ERROR'))

        valid_count = len(predictions) - unclassified_count - error_count
        return valid_count / len(predictions) if predictions else 0

    def _calculate_transferability(self, predictions: List[Dict]) -> float:
        """计算迁移适用性得分 - 分类多样性和有效分类率"""
        if not predictions:
            return 0.0

        categories = set(p.get('category', '') for p in predictions)
        unclassified = sum(1 for p in predictions if '未分类' in p.get('category', ''))

        category_diversity = len(categories) / len(predictions) if predictions else 0
        classified_ratio = (len(predictions) - unclassified) / len(predictions) if predictions else 0

        return (category_diversity + classified_ratio) / 2

    def _calculate_silhouette_score(self, predictions: List[Dict]) -> float:
        """计算轮廓系数作为聚类质量评估指标"""
        if not predictions or len(predictions) < 2:
            return 0.0
        
        # 简单实现：基于类别内距离和类别间距离
        categories = set(p.get('category', '') for p in predictions if p.get('category') and 
                        '未分类' not in p.get('category', '') and 
                        not p.get('category', '').startswith('ERROR'))
        
        if len(categories) <= 1:
            return 0.0
        
        silhouette_scores = []
        
        for pred in predictions:
            cat = pred.get('category', '')
            if cat == '未分类' or cat.startswith('ERROR'):
                continue
            
            content = pred.get('content', '')
            if not content:
                continue
            
            # 计算同类文件的平均距离（相似度）
            same_cat_files = [p for p in predictions if p.get('category') == cat]
            a_i = 0.0
            if len(same_cat_files) > 1:
                distances = []
                for other in same_cat_files:
                    if other != pred:
                        other_content = other.get('content', '')
                        if other_content:
                            # 简单的词重叠距离
                            words1 = set(content.split())
                            words2 = set(other_content.split())
                            if len(words1 | words2) > 0:
                                distance = 1 - len(words1 & words2) / (len(words1 | words2) + 1e-8)
                                distances.append(distance)
                a_i = sum(distances) / len(distances) if distances else 1.0
            
            # 计算到最近其他类别的平均距离
            b_i = float('inf')
            for other_cat in categories:
                if other_cat != cat:
                    other_cat_files = [p for p in predictions if p.get('category') == other_cat]
                    if other_cat_files:
                        distances = []
                        for other in other_cat_files:
                            other_content = other.get('content', '')
                            if other_content:
                                words1 = set(content.split())
                                words2 = set(other_content.split())
                                if len(words1 | words2) > 0:
                                    distance = 1 - len(words1 & words2) / (len(words1 | words2) + 1e-8)
                                    distances.append(distance)
                        if distances:
                            avg_distance = sum(distances) / len(distances)
                            b_i = min(b_i, avg_distance)
            
            if b_i == float('inf'):
                b_i = 1.0
            
            # 轮廓系数
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
                silhouette_scores.append(s_i)
        
        return sum(silhouette_scores) / len(silhouette_scores) if silhouette_scores else 0.0

    def generate_report(self, metrics: Dict, predictions: List[Dict], output_path: str = None) -> str:
        """生成评价报告"""
        report_lines = [
            "=" * 70,
            "分类效果综合评价报告",
            "=" * 70,
            "",
            f"总文件数: {len(predictions)}",
            "",
            "【核心评价指标 - 无监督分类】",
            "-" * 70,
            f"准确率 (Accuracy): {metrics.get('accuracy', 0):.4f}",
            f"纯度 (Purity): {metrics.get('purity', 0):.4f}",
            f"轮廓系数 (Silhouette Score): {metrics.get('silhouette_score', 0):.4f}",
            f"可解释性 (Interpretability): {metrics.get('interpretability', 0):.4f}",
            f"迁移适用性 (Transferability): {metrics.get('transferability', 0):.4f}",
            f"综合得分 (Comprehensive): {metrics.get('comprehensive_score', 0):.4f}",
            "",
            "【分类分布与各类别指标】",
            "-" * 70
        ]

        category_metrics = metrics.get('category_metrics', {})
        for cat, cat_metric in sorted(category_metrics.items(), key=lambda x: -x[1]['count']):
            count = cat_metric['count']
            ratio = cat_metric['ratio'] * 100
            purity = cat_metric['purity']
            report_lines.append(
                f"  {cat}: {count}个 ({ratio:.1f}%) | 纯度:{purity:.3f}"
            )

        report_lines.append("")
        report_lines.append("【分类混淆分析】")
        report_lines.append("-" * 70)
        confusion = metrics.get('confusion_summary', {})
        report_lines.append(f"  有效分类: {confusion.get('valid_classified', 0)}个 ({confusion.get('valid_ratio', 0)*100:.1f}%)")
        report_lines.append(f"  未分类: {confusion.get('unclassified', 0)}个 ({confusion.get('unclassified_ratio', 0)*100:.1f}%)")
        report_lines.append(f"  错误分类: {confusion.get('error', 0)}个")
        report_lines.append(f"  多类别特征: {confusion.get('multi_category_features', 0)}个")

        report_lines.append("")
        report_lines.append("【多类别特征与无法明确归类数据讨论】")
        report_lines.append("-" * 70)
        multi_count = confusion.get('multi_category_features', 0)
        unclass_count = confusion.get('unclassified', 0)
        if multi_count > 0:
            report_lines.append(f"  检测到 {multi_count} 个文件具有多类别特征，建议:")
            report_lines.append(f"    1. 建立主类别+副类别的双重标注机制")
            report_lines.append(f"    2. 对多类别文件进行人工复核确认")
            report_lines.append(f"    3. 考虑引入层次化分类体系")
        if unclass_count > 0:
            report_lines.append(f"  检测到 {unclass_count} 个文件无法明确归类，建议:")
            report_lines.append(f"    1. 扩充分类体系，增加新类别")
            report_lines.append(f"    2. 对未分类文件进行聚类分析，发现潜在新类别")
            report_lines.append(f"    3. 设置'待分类'缓冲区，定期人工审核")

        report = '\n'.join(report_lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report


class PriorityCalculator:
    """优先级计算器 - 基于规则实现，无需AI调用"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.urgency_weight = self.config.get('urgency_weight', 0.4)
        self.misclassification_risk_weight = self.config.get('misclassification_risk_weight', 0.3)
        self.review_necessity_weight = self.config.get('review_necessity_weight', 0.3)

        self.urgency_keywords = [
            '紧急', '急', '限时', '截止', '尽快', '速办', '重要', '优先级',
            '加急', '特急', '立即', '马上', '即刻', '务必', '必须',
            'deadline', 'urgent', 'emergency'
        ]

        self.risk_keywords = [
            '金额', '合同', '协议', '审批', '签字', '盖章', '法律', '资金',
            '预算', '财务', '审计', '合规', '风险', '责任', '赔偿', '违约',
            '诉讼', '仲裁', '处罚', '罚款', '赔偿', '损失'
        ]

        self.funding_keywords = [
            '资金', '预算', '拨款', '经费', '补贴', '补助', '投资', '融资',
            '贷款', '借款', '担保', '抵押', '资产', '财产', '收益', '利润',
            '成本', '费用', '支出', '收入'
        ]

        self.time_sensitive_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}月\d{1,2}日',
            r'截止.*?日',
            r'期限.*?日',
            r'到期',
            r'过期'
        ]

    def calculate_priority(self, file_info: Dict) -> Dict:
        """计算文件优先级 - 纯规则实现"""
        content = file_info.get('content', '')
        category = file_info.get('category', 'unknown')

        analysis = self._analyze_file_factors(content, category)

        urgency_score = self._calculate_urgency(content, analysis)
        risk_score = self._calculate_misclassification_risk(category, analysis)
        review_score = self._calculate_review_necessity(content, analysis)

        overall_score = (
            self.urgency_weight * urgency_score +
            self.misclassification_risk_weight * risk_score +
            self.review_necessity_weight * review_score
        )

        priority_level = self._determine_priority_level(overall_score)

        return {
            'urgency_score': round(urgency_score, 4),
            'risk_score': round(risk_score, 4),
            'review_score': round(review_score, 4),
            'overall_score': round(overall_score, 4),
            'priority_level': priority_level,
            'factors': analysis
        }

    def _analyze_file_factors(self, content: str, category: str) -> Dict:
        """分析文件各项因素 - 基于规则"""
        contains_funding = any(kw in content for kw in self.funding_keywords)
        contains_urgency_keywords = any(kw in content for kw in self.urgency_keywords)
        time_sensitive = any(re.search(pattern, content) for pattern in self.time_sensitive_patterns)

        is_ambiguous = self._check_ambiguity(content, category)
        is_multi_category = self._check_multi_category(content)

        key_entities = self._extract_key_entities(content)
        risk_indicators = self._extract_risk_indicators(content)

        return {
            "contains_funding": contains_funding,
            "contains_urgency_keywords": contains_urgency_keywords,
            "is_ambiguous": is_ambiguous,
            "is_multi_category": is_multi_category,
            "time_sensitive": time_sensitive,
            "key_entities": key_entities[:10],
            "risk_indicators": risk_indicators[:10]
        }

    def _check_ambiguity(self, content: str, category: str) -> bool:
        """检查是否归属不明确"""
        ambiguous_indicators = ['其他', '综合', '通用', '杂项', '未分类', '待分类']
        if any(ind in category for ind in ambiguous_indicators):
            return True

        if len(content) < 50:
            return True

        return False

    def _check_multi_category(self, content: str) -> bool:
        """检查是否有多类别特征"""
        category_indicators = {
            '人事': ['人事', '员工', '招聘', '培训', '考核', '薪资'],
            '财务': ['财务', '资金', '预算', '报销', '审计', '账目'],
            '技术': ['技术', '研发', '开发', '系统', '软件', '硬件'],
            '行政': ['行政', '办公', '后勤', '物业', '设备'],
            '法务': ['法律', '合同', '协议', '诉讼', '合规']
        }

        matched_categories = []
        for cat_name, keywords in category_indicators.items():
            if any(kw in content for kw in keywords):
                matched_categories.append(cat_name)

        return len(matched_categories) >= 2

    def _extract_key_entities(self, content: str) -> List[str]:
        """提取关键实体"""
        entities = []

        org_patterns = [
            r'([\u4e00-\u9fa5]{2,10}(?:公司|局|院|所|中心|部|委|办|厅|处|科|室))',
            r'([\u4e00-\u9fa5]{2,10}(?:大学|学院|学校|医院|银行|企业|集团))'
        ]

        for pattern in org_patterns:
            matches = re.findall(pattern, content)
            entities.extend(matches[:5])

        return list(set(entities))[:10]

    def _extract_risk_indicators(self, content: str) -> List[str]:
        """提取风险指标"""
        indicators = []

        for kw in self.risk_keywords:
            if kw in content:
                indicators.append(kw)

        for kw in self.funding_keywords:
            if kw in content and kw not in indicators:
                indicators.append(f"资金相关:{kw}")

        return indicators[:10]

    def _calculate_urgency(self, content: str, analysis: Dict) -> float:
        """计算紧急程度得分"""
        score = 0.0

        urgency_count = sum(1 for kw in self.urgency_keywords if kw in content)
        score += min(urgency_count * 0.2, 0.6)

        if analysis.get('time_sensitive'):
            score += 0.4

        if analysis.get('contains_urgency_keywords'):
            score += 0.3

        return min(score, 1.0)

    def _calculate_misclassification_risk(self, category: str, analysis: Dict) -> float:
        """计算错分风险得分"""
        score = 0.0

        if analysis.get('is_ambiguous'):
            score += 0.5

        if analysis.get('is_multi_category'):
            score += 0.4

        if '其他' in category or '未分类' in category or category.startswith('ERROR'):
            score += 0.5

        return min(score, 1.0)

    def _calculate_review_necessity(self, content: str, analysis: Dict) -> float:
        """计算复核必要性得分"""
        score = 0.0

        if analysis.get('contains_funding'):
            score += 0.5

        risk_count = sum(1 for kw in self.risk_keywords if kw in content)
        score += min(risk_count * 0.15, 0.5)

        if analysis.get('risk_indicators'):
            score += 0.3

        return min(score, 1.0)

    def _determine_priority_level(self, score: float) -> str:
        """根据得分确定优先级"""
        if score >= 0.6:
            return '高'
        elif score >= 0.35:
            return '中'
        else:
            return '低'


class ReviewDecisionMaker:
    """人工复核决策器"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.review_threshold = self.config.get('review_threshold', 0.5)

    def should_review(self, priority_result: Dict, category: str = None) -> bool:
        """判断是否需要人工复核"""
        if priority_result['priority_level'] == '高':
            return True

        if priority_result['overall_score'] >= self.review_threshold:
            return True

        if category and ('未分类' in category or 'ERROR' in category):
            return True

        return False

    def create_review_plan(self, files_with_priority: List[Dict]) -> Dict:
        """创建人工复核计划 - 包含优先顺序"""
        review_list = []
        auto_process_list = []

        sorted_files = sorted(
            files_with_priority,
            key=lambda x: (-x['priority']['overall_score'], x['file_name'])
        )

        for item in sorted_files:
            if self.should_review(item['priority'], item.get('category')):
                review_list.append(item)
            else:
                auto_process_list.append(item)

        for idx, item in enumerate(review_list, 1):
            item['review_order'] = idx
            item['review_reason'] = self._generate_review_reason(item)

        return {
            'review_list': review_list,
            'auto_process_list': auto_process_list,
            'total_review_count': len(review_list),
            'total_auto_count': len(auto_process_list),
            'review_priority_order': [
                {
                    'order': item['review_order'],
                    'file_name': item['file_name'],
                    'category': item.get('category', ''),
                    'overall_score': item['priority']['overall_score'],
                    'priority_level': item['priority']['priority_level'],
                    'review_reason': item.get('review_reason', '')
                }
                for item in review_list
            ]
        }

    def _generate_review_reason(self, item: Dict) -> str:
        """生成复核原因"""
        reasons = []
        priority = item.get('priority', {})
        category = item.get('category', '')
        
        if priority.get('urgency_score', 0) >= 0.5:
            reasons.append('时效要求较高')
        if priority.get('risk_score', 0) >= 0.5:
            reasons.append('主题归属不明确')
        if priority.get('review_score', 0) >= 0.5:
            reasons.append('涉及资金分配')
        if '未分类' in category or 'ERROR' in category:
            reasons.append('分类不明确')
        if priority.get('priority_level') == '高':
            reasons.append('高优先级')
        
        return '、'.join(reasons) if reasons else '综合评分较高'


class ResourceScenarioAnalyzer:
    """资源约束场景分析器"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.resource_scenarios = {
            'scenario_1': {
                'name': '充足资源',
                'review_capacity': 100,
                'time_limit': None,
                'cost_limit': None
            },
            'scenario_2': {
                'name': '中等资源',
                'review_capacity': 50,
                'time_limit': 48,
                'cost_limit': 1000
            },
            'scenario_3': {
                'name': '有限资源',
                'review_capacity': 20,
                'time_limit': 24,
                'cost_limit': 500
            }
        }

    def load_resource_constraints(self, constraints_file: str) -> Dict:
        """从文件加载资源约束"""
        if not Path(constraints_file).exists():
            return self.resource_scenarios

        try:
            import openpyxl
            wb = openpyxl.load_workbook(constraints_file)
            sheet = wb.active

            scenarios = {}
            for idx, row in enumerate(sheet.iter_rows(values_only=True), 1):
                if idx == 1:
                    continue
                if row[0]:
                    scenarios[f'scenario_{idx - 1}'] = {
                        'name': row[0],
                        'review_capacity': int(row[1]) if row[1] else None,
                        'time_limit': int(row[2]) if row[2] else None,
                        'cost_limit': float(row[3]) if row[3] else None
                    }

            if scenarios:
                self.resource_scenarios.update(scenarios)
        except Exception:
            pass

        return self.resource_scenarios

    def analyze_scenario(self, review_list: List[Dict], scenario_name: str) -> Dict:
        """分析特定资源场景下的处理结果"""
        scenario = self.resource_scenarios.get(scenario_name, {})

        capacity = scenario.get('review_capacity')
        time_limit = scenario.get('time_limit')
        cost_limit = scenario.get('cost_limit')

        if capacity:
            selected_for_review = review_list[:capacity]
            deferred = review_list[capacity:]
        else:
            selected_for_review = review_list
            deferred = []

        # 基于行业标准的成本估算
        # 参考：人工审核文档平均3-8分钟/份，简单文件约2-5分钟
        # 人工成本按50-100元/小时计算
        review_time_per_file = 0.08  # 约5分钟/份（0.08小时）
        review_cost_per_file = 5     # 约5元/份（按60元/小时计算）
        
        total_review_cost = len(selected_for_review) * review_cost_per_file
        total_time_hours = len(selected_for_review) * review_time_per_file

        cost_efficiency = 0
        if cost_limit and cost_limit > 0:
            cost_efficiency = total_review_cost / cost_limit

        time_efficiency = 0
        if time_limit and time_limit > 0:
            time_efficiency = total_time_hours / time_limit

        return {
            'scenario': scenario.get('name', scenario_name),
            'capacity': capacity,
            'time_limit': time_limit,
            'cost_limit': cost_limit,
            'selected_for_review': selected_for_review,
            'deferred_count': len(deferred),
            'deferred_files': deferred[:10],
            'economic_analysis': {
                'total_review_cost': total_review_cost,
                'cost_per_file': review_cost_per_file,
                'total_time_hours': round(total_time_hours, 2),
                'time_per_file_minutes': round(review_time_per_file * 60, 1),
                'cost_efficiency': round(cost_efficiency, 4),
                'time_efficiency': round(time_efficiency, 4),
                'within_budget': total_review_cost <= cost_limit if cost_limit else True,
                'within_time': total_time_hours <= time_limit if time_limit else True
            }
        }

    def compare_scenarios(self, review_list: List[Dict]) -> Dict:
        """比较不同资源场景的处理结果"""
        results = {}

        for scenario_key in self.resource_scenarios:
            results[scenario_key] = self.analyze_scenario(review_list, scenario_key)

        return results

    def generate_recommendations(self, scenario_results: Dict) -> List[str]:
        """生成场景化建议"""
        recommendations = []

        for scenario_key, result in scenario_results.items():
            scenario_name = result['scenario']
            capacity = result['capacity']
            deferred = result['deferred_count']
            economic = result.get('economic_analysis', {})

            if deferred > 0:
                total = deferred + len(result['selected_for_review'])
                ratio = deferred / total if total > 0 else 0
                if ratio > 0.5:
                    recommendations.append(
                        f"{scenario_name}: 资源严重不足（{ratio*100:.0f}%文件延后），建议增加复核人员或延长处理时间"
                    )
                elif ratio > 0.2:
                    recommendations.append(
                        f"{scenario_name}: 可考虑使用AI辅助复核，提高处理效率"
                    )

            if capacity and len(result['selected_for_review']) > 0:
                high_priority = sum(1 for f in result['selected_for_review'] if f['priority']['priority_level'] == '高')
                recommendations.append(
                    f"{scenario_name}: 高优先级文件{high_priority}个，将优先处理"
                )

            if economic:
                if not economic.get('within_budget', True):
                    recommendations.append(
                        f"{scenario_name}: 超出预算（预估{economic['total_review_cost']}元，限额{economic.get('cost_limit', 0)}元），建议优化复核流程"
                    )
                if not economic.get('within_time', True):
                    recommendations.append(
                        f"{scenario_name}: 超出时间限制（预估{economic['total_time_hours']}小时，限额{economic.get('time_limit', 0)}小时），建议增加人手"
                    )
                if economic.get('within_budget', True) and economic.get('within_time', True):
                    recommendations.append(
                        f"{scenario_name}: 经济成本可控（{economic['total_review_cost']}元，{economic['total_time_hours']}小时），方案可行"
                    )

        return recommendations


class Problem3Solver:
    """问题三：评价指标设计、优先级划分和人工复核判断 - 纯规则实现"""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self._script_dir = Path(__file__).parent.parent

        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {}

        self.priority_config = self.config.get('priority', {})
        self.evaluation_config = self.config.get('evaluation', {})

        self.evaluation_metrics = EvaluationMetrics(self.evaluation_config)
        self.priority_calculator = PriorityCalculator(self.priority_config)
        self.review_maker = ReviewDecisionMaker()
        self.resource_analyzer = ResourceScenarioAnalyzer()

    def solve(self, problem2_results: Dict = None,
              problem2_result_path: str = None,
              dataset4_path: str = None,
              output_dir: str = 'output/问题三结果') -> Dict:
        """执行问题三求解 - 无需AI调用"""
        print("=" * 70)
        print("Problem 3: Evaluation, Priority & Review (纯规则实现)")
        print("=" * 70)
        print()

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        files_to_process = self._load_problem2_results(problem2_results, problem2_result_path)

        if not files_to_process:
            return {'success': False, 'error': '未找到problem2分类结果'}

        print(f"待处理文件数: {len(files_to_process)}")
        print()

        print("=" * 70)
        print("第一步：设计评价指标并评估分类效果")
        print("=" * 70)
        evaluation_metrics = self.evaluation_metrics.calculate_comprehensive_metrics(files_to_process)

        print(f"准确率 (Accuracy): {evaluation_metrics.get('accuracy', 0):.4f}")
        print(f"纯度 (Purity): {evaluation_metrics.get('purity', 0):.4f}")
        print(f"轮廓系数 (Silhouette): {evaluation_metrics.get('silhouette_score', 0):.4f}")
        print(f"可解释性 (Interpretability): {evaluation_metrics.get('interpretability', 0):.4f}")
        print(f"迁移适用性 (Transferability): {evaluation_metrics.get('transferability', 0):.4f}")
        print(f"综合得分 (Comprehensive): {evaluation_metrics.get('comprehensive_score', 0):.4f}")
        print()

        evaluation_report_path = str(Path(output_dir) / '评价报告.txt')
        self.evaluation_metrics.generate_report(evaluation_metrics, files_to_process, evaluation_report_path)
        print(f"评价报告已保存: {evaluation_report_path}")
        print()

        print("=" * 70)
        print("第二步：从紧急程度、错分风险、复核必要性三方面划分优先级")
        print("=" * 70)

        if dataset4_path:
            print("加载资源约束...")
            self.resource_analyzer.load_resource_constraints(dataset4_path)

        print("计算文件优先级...")
        prioritized_files = self._calculate_priorities(files_to_process)
        print()

        print("创建人工复核计划...")
        review_plan = self.review_maker.create_review_plan(prioritized_files)
        print(f"  需要人工复核: {review_plan['total_review_count']}个")
        print(f"  自动处理: {review_plan['total_auto_count']}个")
        print()

        print("=" * 70)
        print("第三步：资源约束场景分析")
        print("=" * 70)
        scenario_results = self.resource_analyzer.compare_scenarios(review_plan['review_list'])

        priority_distribution = self._calculate_priority_distribution(prioritized_files)
        recommendations = self.resource_analyzer.generate_recommendations(scenario_results)

        self._save_results({
            'evaluation_metrics': evaluation_metrics,
            'prioritized_files': prioritized_files,
            'review_plan': review_plan,
            'scenario_results': scenario_results,
            'priority_distribution': priority_distribution,
            'recommendations': recommendations
        }, output_dir)

        summary = {
            'success': True,
            'total_files': len(prioritized_files),
            'evaluation_metrics': evaluation_metrics,
            'priority_distribution': priority_distribution,
            'review_plan': {
                'total_review_count': review_plan['total_review_count'],
                'total_auto_count': review_plan['total_auto_count']
            },
            'recommendations': recommendations,
            'output_dir': output_dir
        }

        print()
        print("优先级分布:")
        for level, count in sorted(priority_distribution.items()):
            print(f"  {level}优先级: {count}个")
        print()

        if recommendations:
            print("建议:")
            for rec in recommendations:
                print(f"  - {rec}")
        print()

        print("=" * 70)
        print("Problem 3 Complete")
        print(f"结果文件: {str(Path(output_dir) / '问题三结果.json')}")
        print("=" * 70)

        return summary

    def _load_problem2_results(self, problem2_results: Dict = None,
                                problem2_result_path: str = None) -> List[Dict]:
        """加载problem2的分类结果"""
        files_to_process = []

        if problem2_results and 'results' in problem2_results:
            for file_path, info in problem2_results['results'].items():
                files_to_process.append({
                    'file_path': file_path,
                    'category': info.get('category', '未分类'),
                    'dataset': info.get('dataset', 'unknown'),
                    'extension': info.get('extension', ''),
                    'content': self._read_file_content(file_path)
                })
            return files_to_process

        if not problem2_result_path:
            problem2_result_path = 'output/问题二结果/分类结果.json'

        result_path = Path(problem2_result_path)
        if not result_path.exists():
            result_path = self._script_dir / problem2_result_path

        if not result_path.exists():
            print(f"错误：未找到problem2分类结果文件: {problem2_result_path}")
            return []

        print(f"加载problem2分类结果: {result_path}")
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = data.get('results', {})
            for file_path, info in results.items():
                files_to_process.append({
                    'file_path': file_path,
                    'category': info.get('category', '未分类'),
                    'dataset': info.get('dataset', 'unknown'),
                    'extension': info.get('extension', ''),
                    'content': self._read_file_content(file_path)
                })

            print(f"成功加载 {len(files_to_process)} 个文件的分类结果")
        except Exception as e:
            print(f"加载分类结果失败: {e}")

        return files_to_process

    def _read_file_content(self, file_path: str) -> str:
        """读取文件内容用于优先级计算"""
        try:
            path = Path(file_path)
            if not path.exists():
                return ''

            if path.suffix in ['.txt', '.md', '.csv']:
                return path.read_text(encoding='utf-8', errors='ignore')[:3000]

            return ''
        except Exception:
            return ''

    def _calculate_priorities(self, files: List[Dict]) -> List[Dict]:
        """计算所有文件的优先级 - 纯规则实现"""
        prioritized = []

        for idx, file_info in enumerate(files, 1):
            if idx % 50 == 0:
                print(f"  处理进度: {idx}/{len(files)}...")

            priority = self.priority_calculator.calculate_priority(file_info)

            prioritized.append({
                'file_name': file_info.get('file_name', Path(file_info.get('file_path', '')).name),
                'file_path': file_info.get('file_path', ''),
                'category': file_info.get('category', 'unknown'),
                'dataset': file_info.get('dataset', 'unknown'),
                'priority': priority
            })

        return prioritized

    def _calculate_priority_distribution(self, prioritized_files: List[Dict]) -> Dict[str, int]:
        """计算优先级分布"""
        distribution = defaultdict(int)
        for item in prioritized_files:
            level = item['priority']['priority_level']
            distribution[level] += 1
        return dict(distribution)

    def _save_results(self, results: Dict, output_dir: str):
        """保存结果"""
        result_file = str(Path(output_dir) / '问题三结果.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        summary_file = str(Path(output_dir) / '问题三摘要.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("问题三执行摘要\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"总文件数: {len(results.get('prioritized_files', []))}\n\n")

            f.write("评价指标:\n")
            metrics = results.get('evaluation_metrics', {})
            f.write(f"  准确率: {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"  宏F1: {metrics.get('macro_f1', 0):.4f}\n")
            f.write(f"  可解释性: {metrics.get('interpretability', 0):.4f}\n")
            f.write(f"  迁移适用性: {metrics.get('transferability', 0):.4f}\n")
            f.write(f"  综合得分: {metrics.get('comprehensive_score', 0):.4f}\n\n")

            f.write("优先级分布:\n")
            for level, count in sorted(results.get('priority_distribution', {}).items()):
                f.write(f"  {level}优先级: {count}个\n")

            f.write(f"\n人工复核: {results.get('review_plan', {}).get('total_review_count', 0)}个\n")
            f.write(f"自动处理: {results.get('review_plan', {}).get('total_auto_count', 0)}个\n\n")

            f.write("建议:\n")
            for rec in results.get('recommendations', []):
                f.write(f"  - {rec}\n")
