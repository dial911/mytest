#!/usr/bin/env python3
"""
把已经缓存在 ~/.cache/huggingface/hub 的模型「解引用」复制到项目 ./models/。

为什么不直接用 `huggingface_hub.snapshot_download(local_dir=...)`:
  1. 它会先对每个文件做一次 HEAD 请求校验 (哪怕本地 cache 全齐了)。
  2. 国内镜像 hf-mirror.com 对 5 分钟内的 HEAD 请求数有 100k 配额，很容易 429。

这个脚本只从本地 cache 的 snapshots/<commit>/ 复制 (symlink -> blob realpath)，完全离线、零 HTTP 请求。

用法 (项目根目录下):
    uv run python tools/export_hf_cache_to_models.py \
        shibing624/text2vec-base-chinese \
        cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
也可以不指定参数，导出 main.py 会用到的 3 个默认模型。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_REPOS = [
    "shibing624/text2vec-base-chinese",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "moka-ai/m3e-base",
]


def hf_hub_root() -> Path:
    env = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(env).expanduser().resolve() / "hub"


def repo_dir_name(repo_id: str) -> str:
    """HF cache 约定: org__name -> models--org--name"""
    return "models--" + repo_id.replace("/", "--")


def local_target_name(repo_id: str) -> str:
    """导出到 ./models/ 的目录名，保持 org__name"""
    return repo_id.replace("/", "__")


def copy_tree_dereference(src: Path, dst: Path) -> None:
    """cp -R -L: 把 symlink 解析成真实 blob 再复制成普通文件"""
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_root = dst if rel == "." else dst / rel
        out_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            sf = Path(root) / f
            df = out_root / f
            real = sf.resolve() if sf.is_symlink() else sf
            if real.is_dir():
                # 理论上 files 不可能是 dir，但如果是 symlink 指向了 dir，就递归复制
                shutil.copytree(real, df, symlinks=False, dirs_exist_ok=True)
            else:
                shutil.copy2(real, df, follow_symlinks=True)
        for d in dirs:
            sd = Path(root) / d
            dd = out_root / d
            if not sd.is_symlink():
                continue
            real = sd.resolve()
            shutil.copytree(real, dd, symlinks=False, dirs_exist_ok=True)


def find_latest_snapshot(repo_cache_root: Path) -> Path | None:
    snap_dir = repo_cache_root / "snapshots"
    if not snap_dir.is_dir():
        return None
    snapshots = sorted(p for p in snap_dir.iterdir() if p.is_dir())
    if not snapshots:
        return None
    return snapshots[-1]


def export_one(repo_id: str, out_root: Path, hub: Path) -> tuple[bool, str]:
    src_cache = hub / repo_dir_name(repo_id)
    if not src_cache.is_dir():
        return False, f"HF cache 中未找到 {repo_id} (expected {src_cache}). 先联网跑一次 main.py 让模型下到 cache。"
    snap = find_latest_snapshot(src_cache)
    if snap is None:
        return False, f"{repo_id} 缓存不完整 (snapshots/ 目录为空)"

    target = out_root / local_target_name(repo_id)
    copy_tree_dereference(snap, target)

    # 做一个非常弱的完整性校验 (看 config.json 在不在 + 有 model*.safetensors/bin)
    if not (target / "config.json").is_file():
        return False, f"导出目录缺少 config.json，导出可能异常: {target}"

    # 目录大小
    total_kb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1024
    size_mb = total_kb / 1024
    return True, f"{repo_id:50s} -> {target}  ({size_mb:.1f} MB)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repos", nargs="*", default=DEFAULT_REPOS,
                        help="HuggingFace repo id，例如 shibing624/text2vec-base-chinese")
    parser.add_argument("--out", default="./models", help="导出目录 (默认 ./models)")
    args = parser.parse_args()

    hub = hf_hub_root()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"HF hub cache root: {hub}")
    print(f"导出到:            {out_root}")
    print()

    any_fail = False
    for repo in args.repos:
        ok, msg = export_one(repo, out_root, hub)
        status = "✅" if ok else "❌"
        print(f"{status} {msg}")
        if not ok:
            any_fail = True
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
