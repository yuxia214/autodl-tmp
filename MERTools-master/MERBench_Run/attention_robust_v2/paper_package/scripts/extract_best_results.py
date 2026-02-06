#!/usr/bin/env python3
"""
实验结果提取脚本 - 从NPZ结果文件中自动筛选最高准确度的结果

功能：
- 扫描指定目录下所有NPZ结果文件
- 从文件名中解析 f1、acc、val 指标
- 按 test 集分组，选出每个 test 集的最高 acc 结果
- 支持扫描 baseline/v1/v2/v3/v4 的结果目录做对比
- 输出格式化的对比表格（Markdown格式 + 控制台表格）

用法：
    python extract_best_results.py
    python extract_best_results.py --result_dir /path/to/results
    python extract_best_results.py --output results_comparison.md
"""

import argparse
import glob
import os
import re
from collections import defaultdict


def parse_npz_filename(filepath):
    """
    从NPZ文件名中解析实验指标。

    文件名格式示例：
    cv_features:..._model:attention_robust_v2+utt+None_f1:0.7574_acc:0.7557_val:0.6319_1769910916.npz

    Returns:
        dict: {split, model, f1, acc, val, timestamp, filepath} 或 None
    """
    basename = os.path.basename(filepath)

    # 解析 split (cv, test1, test2, test3)
    split_match = re.match(r'^(cv|test1|test2|test3)_', basename)
    if not split_match:
        return None
    split = split_match.group(1)

    # 解析 model 名称
    model_match = re.search(r'model:([^_]+(?:_[^_+]+)*)\+', basename)
    if not model_match:
        return None
    model = model_match.group(1)

    # 解析 f1, acc, val
    f1_match = re.search(r'f1:([\d.]+)', basename)
    acc_match = re.search(r'acc:([\d.]+)', basename)
    val_match = re.search(r'val:([\d.]+)', basename)

    if not (f1_match and acc_match and val_match):
        return None

    f1 = float(f1_match.group(1))
    acc = float(acc_match.group(1))
    val = float(val_match.group(1))

    # 解析 timestamp
    ts_match = re.search(r'_([\d.]+)\.npz$', basename)
    timestamp = float(ts_match.group(1)) if ts_match else 0.0

    return {
        'split': split,
        'model': model,
        'f1': f1,
        'acc': acc,
        'val': val,
        'timestamp': timestamp,
        'filepath': filepath,
    }


def scan_results(search_dirs):
    """
    扫描多个目录下的NPZ结果文件。

    Args:
        search_dirs: 要搜索的目录列表

    Returns:
        list[dict]: 解析后的结果列表
    """
    results = []
    seen_files = set()

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        # 搜索 **/result/*.npz
        patterns = [
            os.path.join(search_dir, '**', 'result', '*.npz'),
            os.path.join(search_dir, 'result', '*.npz'),
        ]
        for pattern in patterns:
            for filepath in glob.glob(pattern, recursive=True):
                # 去重（同一文件可能被多个pattern匹配）
                real_path = os.path.realpath(filepath)
                if real_path in seen_files:
                    continue
                seen_files.add(real_path)

                parsed = parse_npz_filename(filepath)
                if parsed:
                    results.append(parsed)

    return results


def get_best_per_split(results):
    """
    按 split 分组，选出每组中 acc 最高的记录。

    Args:
        results: 解析后的结果列表

    Returns:
        dict: {split: best_record}
    """
    grouped = defaultdict(list)
    for r in results:
        grouped[r['split']].append(r)

    best = {}
    for split, records in grouped.items():
        best[split] = max(records, key=lambda x: x['acc'])

    return best


def get_best_per_model_split(results):
    """
    按 (model, split) 分组，选出每组中 acc 最高的记录。

    Returns:
        dict: {model: {split: best_record}}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r['model']][r['split']].append(r)

    best = {}
    for model, splits in grouped.items():
        best[model] = {}
        for split, records in splits.items():
            best[model][split] = max(records, key=lambda x: x['acc'])

    return best


MODEL_DISPLAY_NAMES = {
    'attention': 'Baseline (v0)',
    'attention+utt': 'Baseline (v0)',
    'attention_robust': 'Robust v1 (Dropout)',
    'attention_robust_v2': 'P-RMF V2 (VAE)',
    'attention_robust_v3': 'Robust v3',
    'attention_robust_v4': 'Robust v4',
    'attention_robust_v5': 'Robust v5',
}

SPLIT_ORDER = ['cv', 'test1', 'test2', 'test3']


def format_comparison_table(best_per_model_split, markdown=True):
    """
    格式化对比表格。

    Args:
        best_per_model_split: {model: {split: best_record}}
        markdown: 是否输出Markdown格式

    Returns:
        str: 格式化的表格字符串
    """
    lines = []

    # 排序模型
    model_order = ['attention', 'attention+utt', 'attention_robust',
                   'attention_robust_v3', 'attention_robust_v4',
                   'attention_robust_v5', 'attention_robust_v2']
    models = [m for m in model_order if m in best_per_model_split]
    # 添加未在预定义顺序中的模型
    for m in sorted(best_per_model_split.keys()):
        if m not in models:
            models.append(m)

    if markdown:
        # Markdown 表格
        header = '| 方法 |'
        separator = '|------|'
        for split in SPLIT_ORDER:
            header += f' {split} F1 | {split} ACC |'
            separator += '-------:|-------:|'
        lines.append(header)
        lines.append(separator)

        for model in models:
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            splits = best_per_model_split[model]

            # 判断是否为最佳模型（V2），加粗
            is_v2 = model == 'attention_robust_v2'
            if is_v2:
                row = f'| **{display_name}** |'
            else:
                row = f'| {display_name} |'

            for split in SPLIT_ORDER:
                if split in splits:
                    f1 = splits[split]['f1']
                    acc = splits[split]['acc']
                    if is_v2:
                        row += f' **{f1:.4f}** | **{acc:.4f}** |'
                    else:
                        row += f' {f1:.4f} | {acc:.4f} |'
                else:
                    row += ' - | - |'
            lines.append(row)
    else:
        # 控制台表格
        col_width = 12
        name_width = 25

        header = f'{"方法":<{name_width}}'
        for split in SPLIT_ORDER:
            header += f'  {split+" F1":>{col_width}}  {split+" ACC":>{col_width}}'
        lines.append(header)
        lines.append('-' * len(header))

        for model in models:
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            splits = best_per_model_split[model]

            row = f'{display_name:<{name_width}}'
            for split in SPLIT_ORDER:
                if split in splits:
                    f1 = splits[split]['f1']
                    acc = splits[split]['acc']
                    row += f'  {f1:>{col_width}.4f}  {acc:>{col_width}.4f}'
                else:
                    row += f'  {"-":>{col_width}}  {"-":>{col_width}}'
            lines.append(row)

    return '\n'.join(lines)


def format_detail_table(best_per_split, title="最佳结果"):
    """格式化单模型的详细结果表格。"""
    lines = [f'\n### {title}\n']
    lines.append('| Split | F1 | ACC | Val | 文件 |')
    lines.append('|-------|---:|----:|----:|------|')

    for split in SPLIT_ORDER:
        if split in best_per_split:
            r = best_per_split[split]
            short_path = os.path.basename(r['filepath'])[:60] + '...'
            lines.append(
                f"| {split} | {r['f1']:.4f} | {r['acc']:.4f} | "
                f"{r['val']:.4f} | `{short_path}` |"
            )

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='从NPZ结果文件中提取最佳实验结果'
    )
    parser.add_argument(
        '--result_dir', type=str, default=None,
        help='指定搜索目录（默认搜索所有已知目录）'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='输出Markdown文件路径'
    )
    args = parser.parse_args()

    # 确定搜索目录
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.result_dir:
        search_dirs = [args.result_dir]
    else:
        search_dirs = [
            # V2 专属输出目录
            os.path.join(base, 'attention_robust_v2', 'outputs'),
            # 共享结果目录
            os.path.join(base, 'saved-trimodal'),
            # MERBench 目录
            os.path.join(os.path.dirname(base), 'MERBench', 'saved-trimodal'),
            os.path.join(os.path.dirname(base), 'MERBench', 'attention_robust_v2', 'outputs'),
            # v1版本目录
            os.path.join(os.path.dirname(base), 'MERBenchv1版本', 'saved-trimodal'),
            os.path.join(os.path.dirname(base), 'MERBenchv1版本', 'saved-robust-trimodal'),
        ]

    print('=' * 80)
    print('实验结果提取工具 - AttentionRobust 系列模型对比')
    print('=' * 80)

    # 扫描所有结果
    all_results = scan_results(search_dirs)
    print(f'\n共扫描到 {len(all_results)} 个结果文件\n')

    if not all_results:
        print('未找到任何NPZ结果文件。请检查搜索目录。')
        print(f'搜索目录: {search_dirs}')
        return

    # 按模型统计
    model_counts = defaultdict(int)
    for r in all_results:
        model_counts[r['model']] += 1

    print('各模型结果数量:')
    for model, count in sorted(model_counts.items()):
        display = MODEL_DISPLAY_NAMES.get(model, model)
        print(f'  {display}: {count} 个结果文件')

    # 按 (model, split) 分组取最佳
    best_per_model_split = get_best_per_model_split(all_results)

    # 输出对比表格
    print('\n' + '=' * 80)
    print('模型对比表格 (最高ACC)')
    print('=' * 80 + '\n')

    # 控制台格式
    console_table = format_comparison_table(best_per_model_split, markdown=False)
    print(console_table)

    # Markdown格式
    md_table = format_comparison_table(best_per_model_split, markdown=True)

    # 输出V2详细结果
    v2_results = [r for r in all_results if r['model'] == 'attention_robust_v2']
    if v2_results:
        v2_best = get_best_per_split(v2_results)
        print('\n' + '=' * 80)
        print('P-RMF V2 最佳结果详情')
        print('=' * 80)
        for split in SPLIT_ORDER:
            if split in v2_best:
                r = v2_best[split]
                print(f'  {split}: F1={r["f1"]:.4f}, ACC={r["acc"]:.4f}, '
                      f'Val={r["val"]:.4f}')

        # 统计V2所有运行的结果范围
        print('\nP-RMF V2 结果范围 (所有运行):')
        v2_grouped = defaultdict(list)
        for r in v2_results:
            v2_grouped[r['split']].append(r)
        for split in SPLIT_ORDER:
            if split in v2_grouped:
                records = v2_grouped[split]
                accs = [r['acc'] for r in records]
                f1s = [r['f1'] for r in records]
                print(f'  {split}: ACC=[{min(accs):.4f}, {max(accs):.4f}], '
                      f'F1=[{min(f1s):.4f}, {max(f1s):.4f}], '
                      f'共{len(records)}次运行')

    # 输出到文件
    if args.output:
        output_lines = [
            '# 实验结果对比\n',
            '## 模型对比表格 (最高ACC)\n',
            md_table,
            '\n',
        ]

        if v2_results:
            v2_detail = format_detail_table(v2_best, 'P-RMF V2 最佳结果详情')
            output_lines.append(v2_detail)

        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))

        print(f'\n结果已保存到: {args.output}')


if __name__ == '__main__':
    main()
