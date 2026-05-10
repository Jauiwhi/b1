from typing import Dict, List, Any
import json
import numpy as np
from collections import Counter


class ClassificationMetrics:
    """分类效果评价指标"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.accuracy_weight = self.config.get('accuracy_weight', 0.4)
        self.interpretability_weight = self.config.get('interpretability_weight', 0.3)
        self.transferability_weight = self.config.get('transferability_weight', 0.3)

    def calculate_accuracy(self, predictions: List[str], ground_truth: List[str]) -> float:
        """计算准确率"""
        if not predictions or not ground_truth:
            return 0.0
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        return correct / len(predictions)

    def calculate_confusion_matrix(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """计算混淆矩阵"""
        all_labels = sorted(set(predictions + ground_truth))
        label_to_idx = {label: i for i, label in enumerate(all_labels)}

        n = len(all_labels)
        matrix = [[0] * n for _ in range(n)]

        for pred, true in zip(predictions, ground_truth):
            matrix[label_to_idx[true]][label_to_idx[pred]] += 1

        return {
            'matrix': matrix,
            'labels': all_labels,
            'label_to_idx': label_to_idx
        }

    def calculate_precision_recall(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """计算精确率和召回率"""
        confusion = self.calculate_confusion_matrix(predictions, ground_truth)
        matrix = confusion['matrix']
        labels = confusion['labels']
        label_to_idx = confusion['label_to_idx']

        results = {}
        for label in labels:
            idx = label_to_idx[label]

            tp = matrix[idx][idx]
            fp = sum(matrix[r][idx] for r in range(len(labels))) - tp
            fn = sum(matrix[idx][c] for c in range(len(labels))) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': sum(1 for g in ground_truth if g == label)
            }

        return results

    def calculate_macro_f1(self, predictions: List[str], ground_truth: List[str]) -> float:
        """计算宏平均F1分数"""
        pr = self.calculate_precision_recall(predictions, ground_truth)
        if not pr:
            return 0.0
        return sum(v['f1'] for v in pr.values()) / len(pr)

    def calculate_interpretability(self, categories: List[str], descriptions: Dict[str, str] = None) -> float:
        """计算可解释性得分"""
        if not categories:
            return 0.0

        category_counts = Counter(categories)
        n_categories = len(category_counts)

        if n_categories == 1:
            return 0.3

        distribution_entropy = 0
        total = len(categories)
        for count in category_counts.values():
            if count > 0:
                p = count / total
                distribution_entropy -= p * np.log2(p + 1e-10)

        max_entropy = np.log2(n_categories)
        normalized_entropy = distribution_entropy / max_entropy if max_entropy > 0 else 0

        category_name_length_score = 0
        if descriptions:
            for cat in category_counts.keys():
                desc = descriptions.get(cat, '')
                if len(desc) >= 20:
                    category_name_length_score += 1
            category_name_length_score /= n_categories
        else:
            category_name_length_score = 0.5

        interpretability = 0.5 * normalized_entropy + 0.5 * category_name_length_score
        return min(interpretability, 1.0)

    def calculate_transferability(self, training_categories: List[str],
                                  test_predictions: List[str],
                                  test_ground_truth: List[str]) -> float:
        """计算迁移适用性"""
        if not training_categories or not test_predictions:
            return 0.0

        train_cats = set(training_categories)
        unseen_in_test = 0
        for pred in test_predictions:
            if pred not in train_cats:
                unseen_in_test += 1

        known_ratio = 1 - (unseen_in_test / len(test_predictions)) if test_predictions else 0

        accuracy = self.calculate_accuracy(test_predictions, test_ground_truth)

        transfer_score = 0.6 * known_ratio + 0.4 * accuracy
        return min(transfer_score, 1.0)

    def calculate_comprehensive_score(self,
                                       predictions: List[str],
                                       ground_truth: List[str],
                                       descriptions: Dict[str, str] = None,
                                       training_categories: List[str] = None) -> Dict:
        """计算综合评价得分"""
        accuracy = self.calculate_accuracy(predictions, ground_truth)
        macro_f1 = self.calculate_macro_f1(predictions, ground_truth)
        interpretability = self.calculate_interpretability(
            list(set(predictions + ground_truth)), descriptions
        )

        transferability = 0.5
        if training_categories:
            transferability = self.calculate_transferability(
                training_categories, predictions, ground_truth
            )

        comprehensive = (
            self.accuracy_weight * accuracy +
            self.interpretability_weight * interpretability +
            self.transferability_weight * transferability
        )

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'interpretability': interpretability,
            'transferability': transferability,
            'comprehensive_score': comprehensive,
            'precision_recall': self.calculate_precision_recall(predictions, ground_truth)
        }


class MultiCategoryHandler:
    """处理多类别和模糊归属的文件"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)

    def analyze_multi_category(self, file_content: str,
                              model,
                              category_list: List[str]) -> Dict:
        """分析文件是否具有多类别特征"""
        prompt = f"""分析以下文件内容，判断它是否属于多个类别。

可用类别列表：
{chr(10).join(f'- {cat}' for cat in category_list)}

文件内容：
{file_content[:5000]}

请分析并返回JSON格式：
{{
    "is_multi_category": true/false,
    "primary_category": "主要类别",
    "secondary_categories": ["次要类别1", "次要类别2"],
    "confidence": 0.0-1.0,
    "reason": "判断理由"
}}"""

        try:
            response = model.chat([{"role": "user", "content": prompt}])
            result = json.loads(response)
            return result
        except Exception as e:
            return {
                "is_multi_category": False,
                "primary_category": "未分类",
                "secondary_categories": [],
                "confidence": 0.0,
                "reason": f"分析失败: {str(e)}"
            }

    def handle_ambiguous_files(self, ambiguous_files: List[Dict],
                               model,
                               category_list: List[str]) -> List[Dict]:
        """处理归属不明确的文件"""
        results = []

        for file_info in ambiguous_files:
            analysis = self.analyze_multi_category(
                file_info.get('content', ''),
                model,
                category_list
            )

            if analysis['confidence'] >= self.confidence_threshold:
                file_info['assigned_category'] = analysis['primary_category']
                file_info['alternative_categories'] = analysis['secondary_categories']
                file_info['confidence'] = analysis['confidence']
                file_info['reason'] = analysis['reason']
            else:
                file_info['assigned_category'] = '需要人工复核'
                file_info['confidence'] = analysis['confidence']
                file_info['reason'] = analysis['reason']

            results.append(file_info)

        return results


class EvaluationFactory:
    """评价指标工厂"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.classification_metrics = ClassificationMetrics(self.config)
        self.multi_category_handler = MultiCategoryHandler(self.config)

    def evaluate_classification(self,
                                predictions: List[str],
                                ground_truth: List[str],
                                descriptions: Dict[str, str] = None,
                                training_categories: List[str] = None) -> Dict:
        """评价分类效果"""
        return self.classification_metrics.calculate_comprehensive_score(
            predictions, ground_truth, descriptions, training_categories
        )

    def evaluate_new_data(self, predictions: List[str],
                          ground_truth: List[str],
                          descriptions: Dict[str, str] = None) -> Dict:
        """评价新数据归属判断"""
        return self.classification_metrics.calculate_comprehensive_score(
            predictions, ground_truth, descriptions
        )
