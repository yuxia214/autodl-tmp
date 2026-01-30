#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AttentionRobustV2 安装脚本

功能:
1. 将 variational_encoder.py 复制到 toolkit/models/modules/
2. 将 attention_robust_v2.py 复制到 toolkit/models/
3. 更新 toolkit/models/__init__.py 注册新模型
4. 更新 toolkit/data/__init__.py 添加数据集映射

云端使用方法:
    cd /path/to/MERBench
    python attention_robust_v2/setup_v2.py
"""

import os
import shutil
import sys

# 获取脚本所在目录和MERBench根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MERBENCH_ROOT = os.path.dirname(SCRIPT_DIR)  # 假设脚本在 MERBench/attention_robust_v2/

# 目标路径
TOOLKIT_MODELS = os.path.join(MERBENCH_ROOT, 'toolkit', 'models')
TOOLKIT_MODULES = os.path.join(TOOLKIT_MODELS, 'modules')
TOOLKIT_DATA = os.path.join(MERBENCH_ROOT, 'toolkit', 'data')

def check_paths():
    """检查路径是否正确"""
    print("=" * 60)
    print("路径检查")
    print("=" * 60)
    print(f"脚本目录: {SCRIPT_DIR}")
    print(f"MERBench根目录: {MERBENCH_ROOT}")
    print(f"toolkit/models: {TOOLKIT_MODELS}")
    print(f"toolkit/models/modules: {TOOLKIT_MODULES}")
    print(f"toolkit/data: {TOOLKIT_DATA}")
    
    if not os.path.exists(TOOLKIT_MODELS):
        print(f"\n错误: toolkit/models 不存在!")
        print(f"请确保脚本放置在 MERBench/attention_robust_v2/ 目录下")
        return False
    
    if not os.path.exists(TOOLKIT_MODULES):
        print(f"\n警告: toolkit/models/modules 不存在，将创建...")
        os.makedirs(TOOLKIT_MODULES, exist_ok=True)
    
    return True


def copy_files():
    """复制模型文件"""
    print("\n" + "=" * 60)
    print("复制模型文件")
    print("=" * 60)
    
    # 复制 variational_encoder.py
    src_var = os.path.join(SCRIPT_DIR, 'modules', 'variational_encoder.py')
    dst_var = os.path.join(TOOLKIT_MODULES, 'variational_encoder.py')
    if os.path.exists(src_var):
        shutil.copy2(src_var, dst_var)
        print(f"✓ 复制 variational_encoder.py -> {dst_var}")
    else:
        print(f"✗ 未找到 {src_var}")
        return False
    
    # 复制 attention_robust_v2.py
    src_model = os.path.join(SCRIPT_DIR, 'attention_robust_v2.py')
    dst_model = os.path.join(TOOLKIT_MODELS, 'attention_robust_v2.py')
    if os.path.exists(src_model):
        shutil.copy2(src_model, dst_model)
        print(f"✓ 复制 attention_robust_v2.py -> {dst_model}")
    else:
        print(f"✗ 未找到 {src_model}")
        return False
    
    return True


def update_model_init():
    """更新 toolkit/models/__init__.py"""
    print("\n" + "=" * 60)
    print("更新 toolkit/models/__init__.py")
    print("=" * 60)
    
    init_path = os.path.join(TOOLKIT_MODELS, '__init__.py')
    if not os.path.exists(init_path):
        print(f"✗ 未找到 {init_path}")
        return False
    
    with open(init_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经添加
    if 'attention_robust_v2' in content:
        print("✓ attention_robust_v2 已在 __init__.py 中注册")
        return True
    
    # 添加导入
    import_line = "from .attention_robust_v2 import AttentionRobustV2"
    map_line = "    'attention_robust_v2': AttentionRobustV2,"
    
    # 在适当位置添加导入
    if 'from .attention_robust import' in content:
        content = content.replace(
            'from .attention_robust import AttentionRobust',
            'from .attention_robust import AttentionRobust\n' + import_line
        )
    else:
        # 在MODEL_MAP之前添加
        content = content.replace(
            'MODEL_MAP = {',
            import_line + '\n\nMODEL_MAP = {'
        )
    
    # 添加MODEL_MAP条目
    if "'attention_robust':" in content:
        content = content.replace(
            "'attention_robust': AttentionRobust,",
            "'attention_robust': AttentionRobust,\n" + map_line
        )
    else:
        content = content.replace(
            "MODEL_MAP = {",
            "MODEL_MAP = {\n" + map_line
        )
    
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 已更新 {init_path}")
    return True


def update_data_init():
    """更新 toolkit/data/__init__.py"""
    print("\n" + "=" * 60)
    print("更新 toolkit/data/__init__.py")
    print("=" * 60)
    
    init_path = os.path.join(TOOLKIT_DATA, '__init__.py')
    if not os.path.exists(init_path):
        print(f"✗ 未找到 {init_path}")
        return False
    
    with open(init_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'attention_robust_v2' in content:
        print("✓ attention_robust_v2 已在 data/__init__.py 中注册")
        return True
    
    # 添加数据集映射
    map_line = "    'attention_robust_v2': Data_Feat,"
    
    if "'attention_robust':" in content:
        content = content.replace(
            "'attention_robust': Data_Feat,",
            "'attention_robust': Data_Feat,\n" + map_line
        )
    
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 已更新 {init_path}")
    return True


def update_modules_init():
    """更新 toolkit/models/modules/__init__.py"""
    print("\n" + "=" * 60)
    print("更新 toolkit/models/modules/__init__.py")
    print("=" * 60)
    
    init_path = os.path.join(TOOLKIT_MODULES, '__init__.py')
    if not os.path.exists(init_path):
        print(f"✗ 未找到 {init_path}")
        return False
    
    with open(init_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'variational_encoder' in content:
        print("✓ variational_encoder 已在 modules/__init__.py 中注册")
        return True
    
    # 添加导入
    import_lines = """
from .variational_encoder import (
    VariationalMLPEncoder,
    VariationalLSTMEncoder,
    ModalityDecoder,
    UncertaintyWeightedFusion,
    ProxyCrossModalAttention,
    VAELossComputer
)
"""
    content += import_lines
    
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 已更新 {init_path}")
    return True


def verify_installation():
    """验证安装"""
    print("\n" + "=" * 60)
    print("验证安装")
    print("=" * 60)
    
    # 添加MERBench到路径
    sys.path.insert(0, MERBENCH_ROOT)
    
    try:
        from toolkit.models import get_models
        print("✓ toolkit.models 导入成功")
    except ImportError as e:
        print(f"✗ toolkit.models 导入失败: {e}")
        return False
    
    try:
        from toolkit.models import MODEL_MAP
        if 'attention_robust_v2' in MODEL_MAP:
            print("✓ AttentionRobustV2 已注册到 MODEL_MAP")
        else:
            print("✗ AttentionRobustV2 未在 MODEL_MAP 中")
            return False
    except ImportError as e:
        print(f"✗ MODEL_MAP 导入失败: {e}")
        return False
    
    return True


def main():
    print("=" * 60)
    print("AttentionRobustV2 安装脚本")
    print("=" * 60)
    
    if not check_paths():
        print("\n安装失败: 路径检查不通过")
        sys.exit(1)
    
    if not copy_files():
        print("\n安装失败: 文件复制失败")
        sys.exit(1)
    
    if not update_model_init():
        print("\n安装失败: 模型注册失败")
        sys.exit(1)
    
    if not update_data_init():
        print("\n警告: 数据集映射更新失败，请手动添加")
    
    if not update_modules_init():
        print("\n警告: modules/__init__.py 更新失败，请手动添加")
    
    print("\n" + "=" * 60)
    print("安装完成!")
    print("=" * 60)
    print("\n验证安装...")
    
    if verify_installation():
        print("\n✓ 所有组件安装成功!")
        print("\n使用方法:")
        print("  python main-robust.py --model='attention_robust_v2' --use_vae ...")
    else:
        print("\n警告: 验证失败，请检查错误信息")
    

if __name__ == '__main__':
    main()
