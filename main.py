#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI文档处理Agents系统
支持批量分类和处理办公文件
"""

import sys
import os
from pathlib import Path
import argparse
import yaml

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from models.factory import ModelFactory
from problems.problem1 import Problem1Solver
from problems.problem2 import Problem2Solver
from problems.problem3 import Problem3Solver


class AgentSystem:
    """AI文档处理Agents系统主类"""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        self.model_factory = ModelFactory(config_path)
        self.image_mode = 'ocr'

    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[ERROR] Config file not found: {self.config_path}")
            return {}
        except yaml.YAMLError:
            print(f"[ERROR] Config file format error: {self.config_path}")
            return {}

    def check_model(self) -> dict:
        """检查模型连接状态"""
        return self.model_factory.check_model_connection()

    def print_model_status(self, status: dict):
        """打印模型状态"""
        print("=" * 50)
        print("Model Status")
        print("=" * 50)
        print(f"Provider: {status.get('provider', 'unknown')}")
        print(f"Model: {status.get('model_name', 'unknown')}")
        print(f"Connected: {'Yes' if status.get('connected') else 'No'}")
        print(f"Vision Support: {'Yes' if status.get('supports_vision') else 'No'}")
        print()

    def ask_image_mode(self, status: dict) -> str:
        """询问用户图片识别模式"""
        print("=" * 50)
        print("Image Recognition Mode")
        print("=" * 50)
        print()
        print("Choose image recognition method:")
        print()
        print("  [1] AI Vision - Let AI directly recognize images")
        print("      (Note: Most local models have poor vision capability)")
        print("      (Only recommended for cloud models like GPT-4V, Gemini)")
        print()
        print("  [2] OCR + AI - Extract text via OCR first, then AI classify")
        print("      (Recommended for local models, more accurate)")
        print()
        print("  [3] Skip Images - Only process text files")
        print()
        while True:
            choice = input("Select [1-3]: ").strip()
            if choice == '1':
                return 'vision'
            elif choice == '2':
                return 'ocr'
            elif choice == '3':
                return 'skip'
            else:
                print("Invalid choice, try again.")

    def run_problem1(self, dataset_path: str = None, output_dir: str = None) -> dict:
        """运行问题一"""
        print("\n" + "=" * 50)
        print("Problem 1: Dataset1 Classification")
        print("=" * 50 + "\n")

        solver = Problem1Solver(self.config_path)

        status = self.check_model()
        self.print_model_status(status)

        if not status['connected']:
            return {'success': False, 'error': 'Model connection failed'}

        self.image_mode = self.ask_image_mode(status)

        if not dataset_path:
            dataset_path = self._get_dataset_path("数据集1")
        if not output_dir:
            output_dir = self.config.get('processing', {}).get('output_dir', 'output')
            output_dir = str(Path(output_dir) / '问题一结果')

        if not Path(dataset_path).exists():
            print(f"[ERROR] Dataset path not found: {dataset_path}")
            return {'success': False, 'error': 'Dataset not found'}

        return solver.solve(dataset_path, output_dir, image_mode=self.image_mode)

    def run_problem2(self, dataset2_path: str = None, dataset3_path: str = None,
                     output_dir: str = None, categories: list = None) -> dict:
        """运行问题二"""
        print("\n" + "=" * 50)
        print("Problem 2: Dataset2/3 Processing")
        print("=" * 50 + "\n")

        solver = Problem2Solver(self.config_path)

        status = self.check_model()
        self.print_model_status(status)

        if not status['connected']:
            return {'success': False, 'error': 'Model connection failed'}

        self.image_mode = self.ask_image_mode(status)

        if not dataset2_path:
            dataset2_path = self._get_dataset_path("数据集2")
        if not dataset3_path:
            dataset3_path = self._get_dataset_path("数据集3", is_file=True)
        if not output_dir:
            output_dir = self.config.get('processing', {}).get('output_dir', 'output')
            output_dir = str(Path(output_dir) / '问题二结果')

        if categories:
            solver.set_known_categories(categories)

        return solver.solve(dataset2_path, dataset3_path, output_dir, image_mode=self.image_mode)

    def run_problem3(self, problem2_result: dict = None, dataset4_path: str = None,
                      output_dir: str = None) -> dict:
        """运行问题三"""
        print("\n" + "=" * 50)
        print("Problem 3: Evaluation, Priority & Review")
        print("=" * 50 + "\n")

        solver = Problem3Solver(self.config_path)

        status = self.check_model()
        self.print_model_status(status)

        if not status['connected']:
            return {'success': False, 'error': '模型连接失败'}

        if not dataset4_path:
            dataset4_path = self._get_dataset_path("数据集4")

        if not output_dir:
            output_dir = self.config.get('processing', {}).get('output_dir', 'output')
            output_dir = str(Path(output_dir) / '问题三结果')

        # 使用problem2的分类结果文件路径
        problem2_result_path = None
        if problem2_result and 'result_file' in problem2_result:
            problem2_result_path = problem2_result['result_file']

        return solver.solve(problem2_result, problem2_result_path, dataset4_path, output_dir)

    def run_all(self) -> dict:
        """运行所有问题"""
        print("\n" + "=" * 50)
        print("Running All Problems (1 -> 2 -> 3)")
        print("=" * 50 + "\n")

        status = self.check_model()
        self.print_model_status(status)

        if not status['connected']:
            return {'success': False, 'error': 'Model connection failed'}

        self.image_mode = self.ask_image_mode(status)

        print("\n>>> Step 1: Problem 1 <<<")
        result1 = self.run_problem1()
        if not result1.get('success', False):
            print(f"[ERROR] Problem 1 failed: {result1.get('error')}")
            return result1

        print("\n>>> Step 2: Problem 2 <<<")
        result2 = self.run_problem2(categories=result1.get('categories'))
        if not result2.get('success', False):
            print(f"[ERROR] Problem 2 failed: {result2.get('error')}")
            return result2

        print("\n>>> Step 3: Problem 3 <<<")
        result3 = self.run_problem3(result2)
        if not result3.get('success', False):
            print(f"[ERROR] Problem 3 failed: {result3.get('error')}")
            return result3

        print("\n" + "=" * 50)
        print("All Problems Completed!")
        print("=" * 50)

        return {
            'success': True,
            'problem1': result1,
            'problem2': result2,
            'problem3': result3
        }

    def _get_dataset_path(self, dataset_name: str, is_file: bool = False) -> str:
        """获取数据集路径，优先从input文件夹读取，使用关键词模糊匹配"""
        search_keyword = dataset_name.split('：')[0]

        for base_dir in [SCRIPT_DIR / 'input', SCRIPT_DIR, SCRIPT_DIR / 'data', Path('data')]:
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
            return str(SCRIPT_DIR / 'input' / (search_keyword + '.xlsx'))
        return str(SCRIPT_DIR / 'input' / dataset_name)

    def _load_problem1_result(self) -> dict:
        """加载问题一的结果"""
        result_paths = [
            str(Path('output') / '问题一结果' / '分类结果.json'),
            str(Path('output') / '问题一结果' / 'results.json'),
        ]

        for path in result_paths:
            if Path(path).exists():
                try:
                    import json
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    pass

        return None

    def interactive_menu(self):
        """交互式菜单"""
        while True:
            print("\n" + "=" * 50)
            print("AI Document Processing System")
            print("=" * 50)
            print("\nSelect operation:")
            print("  [1] Problem 1 - Dataset1 Classification")
            print("  [2] Problem 2 - Dataset2/3 Processing")
            print("  [3] Problem 3 - Priority Ranking")
            print("  [4] Run All (1-2-3)")
            print("  [5] Edit config.yaml")
            print("  [0] Exit")
            print()

            choice = input("Select [0-5]: ").strip()

            if choice == '0':
                print("\nGoodbye!")
                break
            elif choice == '1':
                self.run_problem1()
            elif choice == '2':
                self.run_problem2()
            elif choice == '3':
                self.run_problem3()
            elif choice == '4':
                self.run_all()
            elif choice == '5':
                self._edit_config()
            else:
                print("\nInvalid choice")

            input("\nPress Enter to continue...")

    def _edit_config(self):
        """编辑配置文件"""
        try:
            if os.name == 'nt':
                os.system(f'notepad {self.config_path}')
            else:
                os.system(f'vim {self.config_path}')

            self.config = self._load_config()
            print("\nConfig updated")
        except Exception as e:
            print(f"\nFailed to edit config: {e}")


def check_and_run_problem(problem_num: int) -> bool:
    """检查并运行指定问题"""
    try:
        agent = AgentSystem()

        status = agent.check_model()
        if not status['connected']:
            print(f"\n[ERROR] Model connection failed: {status.get('error')}")
            return False

        if problem_num == 1:
            result = agent.run_problem1()
        elif problem_num == 2:
            result = agent.run_problem2()
        elif problem_num == 3:
            result = agent.run_problem3()
        else:
            print(f"[ERROR] Invalid problem number: {problem_num}")
            return False

        return result.get('success', False)

    except Exception as e:
        print(f"\n[ERROR] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI Document Processing System')
    parser.add_argument('--problem', type=str, choices=['1', '2', '3', 'all'],
                       help='Select problem (1, 2, 3, or all)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path')
    parser.add_argument('--dataset1', type=str, help='Dataset1 path')
    parser.add_argument('--dataset2', type=str, help='Dataset2 path')
    parser.add_argument('--dataset3', type=str, help='Dataset3 path')
    parser.add_argument('--dataset4', type=str, help='Dataset4 path')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--image-mode', type=str, choices=['vision', 'ocr', 'skip'], default='ocr',
                       help='Image recognition mode: vision=AI视觉, ocr=OCR+AI, skip=跳过图片')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    agent = AgentSystem(args.config)
    agent.image_mode = args.image_mode

    if args.interactive or not args.problem:
        agent.interactive_menu()
        return

    try:
        if args.problem == '1':
            result = agent.run_problem1(args.dataset1, args.output)
        elif args.problem == '2':
            result = agent.run_problem2(args.dataset2, args.dataset3, args.output)
        elif args.problem == '3':
            result = agent.run_problem3(None, args.dataset4, args.output)
        elif args.problem == 'all':
            result = agent.run_all()
        else:
            print(f"[ERROR] Invalid problem: {args.problem}")
            sys.exit(1)

        if result.get('success'):
            sys.exit(0)
        else:
            print(f"\n[ERROR] Failed: {result.get('error')}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
