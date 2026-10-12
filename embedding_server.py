#!/usr/bin/env python3
"""
本地 Embedding / Rerank REST 服务。

接口设计完全兼容 OpenAI Embeddings API & 常见 rerank 协议，和 test_rag 项目内的模型自动对齐：

  POST /v1/embeddings
    body: {"model": "(ignored)", "input": "string or list[string]", "encoding_format": "float"}
    ->  { "data": [{"embedding": [...], "index": 0}], "model": "...", "usage": {"prompt_tokens":...}}

  POST /v1/rerank
    body: {"model":"(ignored)","query":"...","documents":["doc1","doc2",...], "top_n": 3}
    ->  { "results": [{"index":0, "relevance_score":0.98, "document":"..."}, ...], "meta": {...} }

  GET  /healthz
    ->  {"ok": True, "embedding_model": "...", "rerank_model": "..."}

启动方式 (项目根目录下):
    uv run python embedding_server.py --port 8001

然后 main.py 甚至其他语言的程序都可以直接 POST http://127.0.0.1:8001/v1/embeddings 来拿到向量。
默认优先从 ./models/* 本地路径加载 (离线友好)；找不到再回退到 ~/.cache/huggingface/hub + hub id。
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

# 先 load env，保证 HF_ENDPOINT / 代理等初始化成功
load_dotenv()
if os.getenv("HF_ENDPOINT"):
    os.environ.setdefault("HF_ENDPOINT", os.environ["HF_ENDPOINT"])


DEFAULT_EMBEDDING_REPO = "shibing624/text2vec-base-chinese"
DEFAULT_RERANK_REPO = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _candidate_paths(repo_id: str) -> List[str]:
    cands: List[str] = []
    local = Path("models") / repo_id.replace("/", "__")
    local_abs = local.resolve()
    if local_abs.is_dir() and (local_abs / "config.json").is_file():
        cands.append(str(local_abs))
    cands.append(repo_id)
    return cands


def _load_sentence_transformer(repo: str, trust_remote_code: bool) -> SentenceTransformer:
    last_err: Optional[Exception] = None
    for path in _candidate_paths(repo):
        label = f"本地路径 {path}" if os.path.isdir(path) else f"Hub 仓库 {path}"
        print(f"[embedding_server] 加载 Embedding 模型: {label}")
        try:
            return SentenceTransformer(path, trust_remote_code=trust_remote_code)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  WARN: 加载失败 {type(e).__name__}: {e}")
    raise RuntimeError(f"Embedding 模型加载失败: {repo}. 最后错误: {type(last_err).__name__}: {last_err}")


def _load_cross_encoder(repo: str) -> CrossEncoder:
    last_err: Optional[Exception] = None
    for path in _candidate_paths(repo):
        label = f"本地路径 {path}" if os.path.isdir(path) else f"Hub 仓库 {path}"
        print(f"[embedding_server] 加载 Rerank 模型: {label}")
        try:
            return CrossEncoder(path)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  WARN: 加载失败 {type(e).__name__}: {e}")
    raise RuntimeError(f"Rerank 模型加载失败: {repo}. 最后错误: {type(last_err).__name__}: {last_err}")


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["embedding_model_name"] = os.getenv("EMBEDDING_MODEL_NAME") or DEFAULT_EMBEDDING_REPO
    _state["rerank_model_name"] = os.getenv("RERANK_MODEL_NAME") or DEFAULT_RERANK_REPO
    trust = os.getenv("EMBEDDING_MODEL_TRUST_REMOTE", "false").lower() in ("1", "true", "yes", "on")
    t0 = time.time()
    _state["embedding"] = _load_sentence_transformer(_state["embedding_model_name"], trust)
    _state["rerank"] = _load_cross_encoder(_state["rerank_model_name"])
    print(f"[embedding_server] 两个模型都加载完成，共 {time.time() - t0:.1f}s")
    try:
        yield
    finally:
        _state.clear()


app = FastAPI(title="Local Embedding + Rerank Server", version="1.0.0", lifespan=lifespan)


# ---------------- Pydantic schemas ----------------
class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    encoding_format: str = "float"


class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str
    documents: List[str]
    top_n: int = Field(default=0, ge=0, description="0 表示返回全部 documents")
    return_documents: bool = True


# ---------------- endpoints ----------------
@app.get("/healthz", tags=["meta"])
async def healthz():
    return {
        "ok": True,
        "embedding_model": _state.get("embedding_model_name"),
        "rerank_model": _state.get("rerank_model_name"),
    }


@app.post("/v1/embeddings", tags=["embedding"])
async def create_embeddings(req: EmbeddingsRequest):
    st: SentenceTransformer = _state["embedding"]
    inputs = [req.input] if isinstance(req.input, str) else list(req.input)
    if not inputs:
        raise HTTPException(status_code=400, detail="input 不能为空")
    t0 = time.time()
    try:
        embeds = st.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"encode 失败: {type(e).__name__}: {e}")

    data = []
    dim = 0
    for i, v in enumerate(embeds):
        v_list = v.tolist()
        dim = dim or len(v_list)
        data.append({
            "object": "embedding",
            "embedding": v_list,
            "index": i,
        })
    return {
        "object": "list",
        "data": data,
        "model": _state.get("embedding_model_name"),
        "usage": {
            "prompt_tokens": sum(max(1, len(s)) for s in inputs),
            "total_tokens": sum(max(1, len(s)) for s in inputs),
        },
        "meta": {
            "dimension": dim,
            "count": len(inputs),
            "elapsed_ms": round((time.time() - t0) * 1000),
        },
    }


@app.post("/v1/rerank", tags=["rerank"])
async def rerank(req: RerankRequest):
    ce: CrossEncoder = _state["rerank"]
    if not req.documents:
        raise HTTPException(status_code=400, detail="documents 不能为空")
    pairs = [(req.query, doc) for doc in req.documents]
    t0 = time.time()
    try:
        scores = ce.predict(pairs).tolist()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"rerank 失败: {type(e).__name__}: {e}")

    scored = sorted(
        [{"index": i, "relevance_score": float(s), "document": doc}
         for i, (s, doc) in enumerate(zip(scores, req.documents))],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )
    if req.top_n and req.top_n < len(scored):
        scored = scored[: req.top_n]
    if not req.return_documents:
        for item in scored:
            item.pop("document", None)
    return {
        "results": scored,
        "meta": {
            "model": _state.get("rerank_model_name"),
            "query": req.query,
            "n": len(req.documents),
            "top_n": req.top_n or len(req.documents),
            "elapsed_ms": round((time.time() - t0) * 1000),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("缺少 uvicorn。请先: uv pip install uvicorn[standard] fastapi pydantic")
        raise SystemExit(1)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
