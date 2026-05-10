from typing import Dict, List
import json
from pathlib import Path
from .metrics import EvaluationFactory


class EvaluationEngine:
    """评价引擎，用于评估分类和优先级划分效果"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.factory = EvaluationFactory(self.config)

    def evaluate_classification_effect(self,
                                       predictions: List[Dict],
                                       ground_truth: List[Dict],
                                       category_descriptions: Dict[str, str] = None,
                                       training_categories: List[str] = None) -> Dict:
        """评价分类效果"""
        pred_labels = [p.get('category', 'unknown') for p in predictions]
        true_labels = [g.get('category', 'unknown') for g in ground_truth]

        return self.factory.evaluate_classification(
            pred_labels, true_labels, category_descriptions, training_categories
        )

    def evaluate_new_data_classification(self,
                                         dataset2_predictions: List[Dict],
                                         dataset3_predictions: List[Dict],
                                         ground_truth: Dict = None) -> Dict:
        """评价新数据归属判断"""
        results = {
            'dataset2': {},
            'dataset3': {},
            'combined': {}
        }

        if dataset2_predictions:
            pred_labels = [p.get('category', 'unknown') for p in dataset2_predictions]
            true_labels = [g.get('category', 'unknown') for g in ground_truth.get('dataset2', [])] if ground_truth else pred_labels
            results['dataset2'] = self.factory.evaluate_new_data(pred_labels, true_labels)

        if dataset3_predictions:
            pred_labels = [p.get('category', 'unknown') for p in dataset3_predictions]
            true_labels = [g.get('category', 'unknown') for g in ground_truth.get('dataset3', [])] if ground_truth else pred_labels
            results['dataset3'] = self.factory.evaluate_new_data(pred_labels, true_labels)

        all_preds = [p.get('category', 'unknown') for p in dataset2_predictions + dataset3_predictions]
        all_truths = []
        if ground_truth:
            all_truths = [g.get('category', 'unknown') for g in ground_truth.get('dataset2', []) + ground_truth.get('dataset3', [])]
        else:
            all_truths = all_preds

        results['combined'] = self.factory.evaluate_new_data(all_preds, all_truths)

        return results

    def generate_evaluation_report(self, evaluation_results: Dict, output_path: str = None) -> str:
        """生成评价报告"""
        report_lines = [
            "=" * 60,
            "分类效果综合评价报告",
            "=" * 60,
            ""
        ]

        for dataset_name, metrics in evaluation_results.items():
            if not metrics:
                continue

            report_lines.append(f"【{dataset_name}】")
            report_lines.append("-" * 40)
            report_lines.append(f"准确率 (Accuracy): {metrics.get('accuracy', 0):.4f}")
            report_lines.append(f"宏F1分数 (Macro F1): {metrics.get('macro_f1', 0):.4f}")
            report_lines.append(f"可解释性 (Interpretability): {metrics.get('interpretability', 0):.4f}")
            report_lines.append(f"迁移适用性 (Transferability): {metrics.get('transferability', 0):.4f}")
            report_lines.append(f"综合得分 (Comprehensive): {metrics.get('comprehensive_score', 0):.4f}")
            report_lines.append("")

            pr = metrics.get('precision_recall', {})
            if pr:
                report_lines.append("各分类指标:")
                for cat, values in pr.items():
                    report_lines.append(f"  {cat}:")
                    report_lines.append(f"    Precision: {values['precision']:.4f}")
                    report_lines.append(f"    Recall: {values['recall']:.4f}")
                    report_lines.append(f"    F1: {values['f1']:.4f}")
                    report_lines.append(f"    Support: {values['support']}")
                report_lines.append("")

        report = '\n'.join(report_lines)

        if output_path:
            output_dir = Path(output_path).parent
            if str(output_dir) != '.':
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report
