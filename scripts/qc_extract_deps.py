#!/usr/bin/env python3
"""
极简依赖提取器 — 用正则从代码中"抠"出硬依赖，不读完整文件。

用法:
  python scripts/qc_extract_deps.py [--format json|plain] [path]

输出:
  只输出正则匹配到的依赖行，不输出无关文件内容（省 token）。
"""

import json
import re
import sys
from pathlib import Path

# ── 四类正则模式（针对性提取，不读文件全文） ──────────────

# 1. pyproject.toml / requirements.txt 中的包名
DEP_PATTERN = re.compile(
    r'^[\s\'"]*([a-zA-Z0-9_\-\.]+)'   # 包名
    r'(?:\s*[><=!~]+\s*[\d.*]+)?'      # 可选版本号
    r'(?:\s*[,;]|$)',                   # 分隔或结束
    re.MULTILINE
)

# 2. Python import / from 语句（源码中的硬依赖）
IMPORT_PATTERN = re.compile(
    r'^(?:import|from)\s+([a-zA-Z0-9_\.]+)', re.MULTILINE
)

# 3. sys.path 插入 / PYTHONPATH 引用 — 本地路径硬编码
PATH_PATTERN = re.compile(
    r'(?:sys\.path\.insert|sys\.path\.append|PYTHONPATH|PATH=)'
    r'[\s\(]*[\'"]?([a-zA-Z]:[\\/][^\'")\s]+)[\'"]?',
)

# 4. Conda / pip install 行（Dockerfile, .bat, Makefile 中的）
INSTALL_PATTERN = re.compile(
    r'(?:pip|conda|mamba)\s+install\s+(.+?)(?:&&|\||$)', re.IGNORECASE
)


def scan_pyproject(filepath: str) -> list:
    """从 pyproject.toml 提取依赖（仅读 [project]dependencies 段）"""
    content = Path(filepath).read_text(encoding='utf-8')
    # 只切依赖段，不读整个文件
    m = re.search(r'\[project\].*?dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if not m:
        return []
    raw = m.group(1)
    return [line.strip().strip('"').strip("'").rstrip(',')
            for line in raw.split('\n')
            if line.strip() and not line.strip().startswith('#')]


def scan_imports(filepath: str) -> set:
    """从 .py 文件提取顶层 import（不读函数体内部的 import）"""
    content = Path(filepath).read_text(encoding='utf-8')
    lines = content.split('\n')
    imports = set()
    for line in lines:
        # 跳过注释和空行
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        m = IMPORT_PATTERN.match(stripped)
        if m:
            # 取 import 根包名 torch.utils.data → torch
            root = m.group(1).split('.')[0]
            if root != '__future__':
                imports.add(root)
        # 检测到函数/类定义就停（import 只位于模块顶层）
        if stripped.startswith(('def ', 'class ', '@')):
            break
    return imports


def scan_install_commands(filepath: str) -> list:
    """从 .bat / .sh / Dockerfile 找安装命令中的依赖"""
    content = Path(filepath).read_text(encoding='utf-8', errors='replace')
    found = []
    for m in INSTALL_PATTERN.finditer(content):
        pkgs = re.findall(r'([a-zA-Z0-9_\-\.]+(?:==[\d.*]+)?)', m.group(1))
        found.extend(pkgs)
    return found


def scan_local_paths(filepath: str) -> list:
    """从 .bat / .py 找硬编码的本地路径"""
    content = Path(filepath).read_text(encoding='utf-8', errors='replace')
    return [m.group(1) for m in PATH_PATTERN.finditer(content)]


def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    fmt = 'json' if '--format' in sys.argv and sys.argv[sys.argv.index('--format')+1] == 'json' else 'plain'

    result = {
        'pyproject_deps': [],
        'source_imports': {},
        'install_scripts': [],
        'local_paths': [],
    }

    # 只扫描关键文件（不遍历无关文件）
    for f in sorted(Path(root_dir).rglob('*')):
        if not f.is_file():
            continue
        # 跳过大目录
        skip_dirs = {'__pycache__', '.git', 'venv', '.venv', 'env',
                     'node_modules', 'build', 'dist', '.pytest_cache',
                     '.ruff_cache', '.workbuddy', 'output', 'testset',
                     '.pilot_venv', '.superpowers', '.agents', '.claude'}
        if any(p.name in skip_dirs for p in f.parents):
            continue

        # 跳过二进制和非目标扩展名
        ext = f.suffix.lower()
        if ext == '.py':
            imports = scan_imports(str(f))
            if imports:
                result['source_imports'][str(f.relative_to(root_dir))] = sorted(imports)
        elif ext in ('.bat', '.sh', '.ps1', '.Dockerfile', 'dockerfile'):
            pkgs = scan_install_commands(str(f))
            if pkgs:
                result['install_scripts'].append({
                    'file': str(f.relative_to(root_dir)),
                    'packages': pkgs
                })
            paths = scan_local_paths(str(f))
            if paths:
                result['local_paths'].append({
                    'file': str(f.relative_to(root_dir)),
                    'paths': paths
                })

    # pyproject.toml
    pyproject = Path(root_dir) / 'pyproject.toml'
    if pyproject.exists():
        result['pyproject_deps'] = scan_pyproject(str(pyproject))

    # 输出
    if fmt == 'json':
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"{'='*60}")
        print("📦 项目依赖摘要")
        print(f"{'='*60}")
        if result['pyproject_deps']:
            print(f"\npyproject.toml 声明依赖 ({len(result['pyproject_deps'])} 项):")
            for d in sorted(result['pyproject_deps']):
                print(f"  • {d}")
        for f, imps in sorted(result['source_imports'].items()):
            print(f"\n  {f} 用到 {len(imps)} 个包:")
            for i in sorted(imps):
                print(f"    └─ {i}")
        if result['install_scripts']:
            print("\n📜 安装脚本中的依赖:")
            for s in result['install_scripts']:
                print(f"  {s['file']}: {', '.join(s['packages'])}")
        if result['local_paths']:
            print("\n⚠️ 本地路径硬编码:")
            for s in result['local_paths']:
                for p in s['paths']:
                    print(f"  {s['file']} → {p}")


if __name__ == '__main__':
    main()
