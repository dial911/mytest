import os
from dotenv import load_dotenv

load_dotenv()

HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT


from typing import List


def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, 'r') as file:
        content = file.read()

    return [chunk for chunk in content.split("\n\n")]


chunks = split_into_chunks("doc.md")

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "shibing624/text2vec-base-chinese",
)
EMBEDDING_MODEL_TRUST_REMOTE = os.getenv(
    "EMBEDDING_MODEL_TRUST_REMOTE", "false"
).lower() in ("1", "true", "yes", "on")
HF_HUB_DOWNLOAD_TIMEOUT = int(os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "300"))

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(HF_HUB_DOWNLOAD_TIMEOUT))


def _load_sentence_transformer() -> SentenceTransformer:
    last_error = None
    candidates = [EMBEDDING_MODEL_NAME]
    if EMBEDDING_MODEL_NAME != "shibing624/text2vec-base-chinese":
        candidates.append("shibing624/text2vec-base-chinese")
    candidates.append("moka-ai/m3e-base")

    for model_name in candidates:
        try:
            print(f"[INFO] 加载 Embedding 模型: {model_name}")
            return SentenceTransformer(
                model_name,
                trust_remote_code=EMBEDDING_MODEL_TRUST_REMOTE,
            )
        except Exception as e:
            last_error = e
            print(f"[WARN] 加载 {model_name} 失败: {type(e).__name__}: {e}")

    raise RuntimeError(
        "所有 Embedding 模型均加载失败。请检查：\n"
        "  1. 网络 / 代理是否正常（.env 中取消 HTTPS_PROXY 注释并填写正确端口）\n"
        "  2. 或在 .env 中设置 EMBEDDING_MODEL_NAME 为已下载的本地路径\n"
        f"最后一次错误: {type(last_error).__name__}: {last_error}"
    )


embedding_model = _load_sentence_transformer()


def embed_chunk(chunk: str) -> List[float]:
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()


embeddings = [embed_chunk(chunk) for chunk in chunks]

import chromadb

chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")


def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )


save_embeddings(chunks, embeddings)


def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]


query = "哆啦A梦使用的3个秘密道具分别是什么？"
retrieved_chunks = retrieve(query, 5)

for i, chunk in enumerate(retrieved_chunks):
    print(f"[{i}] {chunk}\n")

from sentence_transformers import CrossEncoder

RERANK_MODEL_NAME = os.getenv(
    "RERANK_MODEL_NAME",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)


def _load_cross_encoder() -> CrossEncoder:
    last_error = None
    candidates = [RERANK_MODEL_NAME]
    if RERANK_MODEL_NAME != "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1":
        candidates.append("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    for model_name in candidates:
        try:
            print(f"[INFO] 加载 Rerank 模型: {model_name}")
            return CrossEncoder(model_name)
        except Exception as e:
            last_error = e
            print(f"[WARN] 加载 {model_name} 失败: {type(e).__name__}: {e}")

    raise RuntimeError(
        "所有 Rerank 模型均加载失败。请检查代理/网络。\n"
        f"最后一次错误: {type(last_error).__name__}: {last_error}"
    )


_cross_encoder = _load_cross_encoder()


def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = _cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]


reranked_chunks = rerank(query, retrieved_chunks, 3)

for i, chunk in enumerate(reranked_chunks):
    print(f"[{i}] {chunk}\n")


import time
import httpx

API_TIMEOUT = int(os.getenv("API_TIMEOUT", "180"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
API_RETRY_DELAY = int(os.getenv("API_RETRY_DELAY", "2"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
LLM_BASE_URL_DEFAULT = {
    "deepseek":    "https://api.deepseek.com/v1",
    "moonshot":    "https://api.moonshot.cn/v1",
    "zhipu":       "https://open.bigmodel.cn/api/paas/v4",
    "siliconflow": "https://api.siliconflow.cn/v1",
}
MODEL_DEFAULTS = {
    "deepseek":    "deepseek-chat",
    "moonshot":    "kimi-k2.6",
    "zhipu":       "glm-4-flash",
    "siliconflow": "deepseek-ai/DeepSeek-V2.5",
}
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or LLM_BASE_URL_DEFAULT.get(LLM_PROVIDER, "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL") or MODEL_DEFAULTS.get(LLM_PROVIDER, "deepseek-chat")

LLM_TEMPERATURE_RAW = os.getenv("LLM_TEMPERATURE", "")
if LLM_TEMPERATURE_RAW == "":
    _MODEL_FORCES_TEMP_1 = LLM_PROVIDER == "moonshot" and LLM_MODEL.startswith(("kimi-k2", "kimi-k"))
    LLM_TEMPERATURE = 1.0 if _MODEL_FORCES_TEMP_1 else 0.2
else:
    LLM_TEMPERATURE = float(LLM_TEMPERATURE_RAW)

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

_COMPATIBLE_SUGGESTIONS = {
    "deepseek":    "在 .env 中设置 DEEPSEEK_API_KEY=sk-xxx（https://platform.deepseek.com）",
    "moonshot":    "在 .env 中设置 LLM_BASE_URL=https://api.moonshot.cn/v1 LLM_MODEL=kimi-k2.6 DEEPSEEK_API_KEY=sk-xxx （https://platform.moonshot.cn）",
    "zhipu":       "在 .env 中设置 LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4 LLM_MODEL=glm-4-flash DEEPSEEK_API_KEY=xxx",
    "siliconflow": "在 .env 中设置 LLM_BASE_URL=https://api.siliconflow.cn/v1 LLM_MODEL=deepseek-ai/DeepSeek-V2.5 DEEPSEEK_API_KEY=sk-xxx",
    "openai-like": "任何 OpenAI 兼容接口，都只要配 LLM_BASE_URL / LLM_MODEL / DEEPSEEK_API_KEY 即可",
}

PRESET_BY_BASE_DOMAIN = {
    "api.deepseek.com":              ("deepseek",    "deepseek-chat"),
    "api.moonshot.cn":               ("moonshot",    "kimi-k2.6"),
    "open.bigmodel.cn":              ("zhipu",       "glm-4-flash"),
    "api.siliconflow.cn":            ("siliconflow", "deepseek-ai/DeepSeek-V2.5"),
}

_EXPECTED_BASE_URL_MODEL: dict = {}
for _p, _m in MODEL_DEFAULTS.items():
    _base = LLM_BASE_URL_DEFAULT.get(_p, "")
    _dom = _base.split("://", 1)[1].split("/", 1)[0].rstrip(".") if "://" in _base else ""
    if _dom:
        _EXPECTED_BASE_URL_MODEL[_dom] = (_p, _m)


def _normalize_provider_base_model(provider: str, base_url: str, model: str):
    """根据 base_url 的域名推断用户是不是把 provider/base/model 搞混了；
    如果发现典型的「base 指向 moonshot.cn 但 model 填了 deepseek-chat」就给出修正提示。"""
    issues: list[str] = []
    dom = ""
    if "://" in base_url:
        dom = base_url.split("://", 1)[1].split("/", 1)[0].rstrip(".")
    expected = _EXPECTED_BASE_URL_MODEL.get(dom)
    if expected:
        exp_provider, exp_model = expected
        if provider != exp_provider:
            issues.append(f"当前 BASE_URL={dom} 属于厂商 {exp_provider!r}，但 LLM_PROVIDER={provider!r}，建议改为 LLM_PROVIDER={exp_provider!r}")
        if model != exp_model and model not in {"deepseek-chat", "moonshot-v1-8k", "moonshot-v1-128k", "moonshot-v1-auto"}:
            pass
        elif model in {"moonshot-v1-8k", "moonshot-v1-auto", "moonshot-v1-128k"} and exp_provider == "moonshot":
            issues.append(f"你的 Moonshot 账号在 /v1/models 下可用的模型可能不包含 {model!r}；如果 HTTP 404 请改用默认 {exp_model!r}（新账号免费额度默认只开 kimi-k2 系列）")
    return issues


_VENDOR_STARTUP_ISSUES = _normalize_provider_base_model(LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL)
if _VENDOR_STARTUP_ISSUES:
    for _i in _VENDOR_STARTUP_ISSUES:
        print(f"[HINT] {_i}")


def _build_httpx_timeout(total_seconds: int) -> httpx.Timeout:
    return httpx.Timeout(
        connect=max(20, min(60, total_seconds // 2)),
        read=total_seconds,
        write=total_seconds,
        pool=max(10, total_seconds // 6),
    )


def _create_llm_http_client() -> httpx.Client:
    client_kwargs: dict = {
        "timeout": _build_httpx_timeout(API_TIMEOUT),
        "follow_redirects": True,
        "http2": True,
    }
    # if PROXY:
    #     client_kwargs["proxy"] = PROXY

    c = httpx.Client(**client_kwargs)

    pool_desc = "unknown"
    try:
        for tpt in c._mounts.values():
            pool = getattr(tpt, "_pool", None)
            if pool is not None:
                pool_desc = type(pool).__name__
                break
    except Exception:
        pass

    base = (LLM_BASE_URL or "").rstrip("/")
    print(f"[INFO] LLM: provider={LLM_PROVIDER!r} model={LLM_MODEL!r} base_url={base!r}")

    if not DEEPSEEK_API_KEY:
        msg = f"[ERROR] 缺少 API Key。{_COMPATIBLE_SUGGESTIONS.get(LLM_PROVIDER, _COMPATIBLE_SUGGESTIONS['openai-like'])}"
        raise RuntimeError(msg)

    return c


llm_http_client = _create_llm_http_client()


def generate(query: str, chunks: List[str]) -> str:
    base = (LLM_BASE_URL or "").rstrip("/")
    url = f"{base}/chat/completions"

    system = "你是一位严谨的中文知识助手。必须严格依据提供的相关片段作答，禁止编造，若片段中无依据就明确回答「根据现有资料无法回答」。"
    user_parts = [
        "用户问题:",
        query,
        "",
        "相关片段:",
    ]
    if chunks:
        user_parts.extend(f"- {c}" for c in chunks)
    else:
        user_parts.append("- (无)")
    user_parts.extend(["", "请回答:"])
    payload = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": "\n".join(user_parts)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    print(f"--- Request POST {url} ---\n"
          f"model={LLM_MODEL}  temperature={LLM_TEMPERATURE}  max_tokens={LLM_MAX_TOKENS}\n"
          f"--- User prompt preview ---\n"
          f"{payload['messages'][1]['content'][:600]}\n"
          f"{'' if len(payload['messages'][1]['content']) <= 600 else '... (truncated)'}\n\n---")

    last_error: Exception | None = None
    transient_exc = (
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.RemoteProtocolError,
        TimeoutError,
        ConnectionError,
    )
    retry_on_http = {408, 429, 500, 502, 503, 504}
    terminal_balance_http = {402}
    terminal_auth_http = {401, 403}
    terminal_client_http = {400, 404, 405, 406, 411, 413, 414, 415, 422} - {408, 429}

    def _is_balance_error(status: int, payload_data: dict) -> bool:
        if status in terminal_balance_http:
            return True
        if status != 400:
            return False
        err_msg = ""
        err_obj = payload_data.get("error") if isinstance(payload_data, dict) else None
        if isinstance(err_obj, dict):
            err_msg = str(err_obj.get("message") or "").lower()
        elif isinstance(err_obj, str):
            err_msg = err_obj.lower()
        return any(k in err_msg for k in ("balance", "insufficient", "quota", "credit", "out of money"))

    def _is_auth_error(status: int, payload_data: dict) -> bool:
        if status in terminal_auth_http:
            return True
        err_obj = payload_data.get("error") if isinstance(payload_data, dict) else None
        if isinstance(err_obj, dict):
            code = str(err_obj.get("code") or err_obj.get("type") or "").lower()
            msg = str(err_obj.get("message") or "").lower()
            if any(k in code for k in ("auth", "unauth", "forbidden", "apikey", "api_key")):
                return True
            if any(k in msg for k in ("api key", "apikey", "unauthorized", "invalid key")):
                return True
        return False

    def _extract_answer(msg: dict, usage_obj: dict) -> str | None:
        content = msg.get("content")
        reasoning = msg.get("reasoning_content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(reasoning, str) and reasoning.strip():
            stripped = reasoning.strip()
            last_non_empty = ""
            for line in reversed(stripped.splitlines()):
                s = line.strip()
                if s:
                    last_non_empty = s
                    break
            return last_non_empty or stripped
        if usage_obj:
            return f"（模型未返回文本内容，usage={usage_obj}）"
        return None

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            print(f"[INFO] 调用 {LLM_PROVIDER.upper()} /chat/completions (第 {attempt}/{API_MAX_RETRIES} 次)...")
            r = llm_http_client.post(url, json=payload, headers=headers)
            status_ok = 200 <= r.status_code < 300
            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text[:500]}

            if status_ok:
                choices = data.get("choices") or []
                usage = data.get("usage") or {}
                if choices:
                    msg0 = choices[0].get("message") or {}
                    ans = _extract_answer(msg0, usage)
                    if ans:
                        return ans
                last_error = RuntimeError(f"HTTP {r.status_code} 但响应结构异常: {data!r}")
                print(f"[WARN] 第 {attempt} 次响应异常: {last_error}")
            else:
                http_err = RuntimeError(f"HTTP {r.status_code}: {data!r}")
                last_error = http_err
                preview = str(data)[:400]
                if _is_balance_error(r.status_code, data):
                    print(f"[FATAL] 账户余额不足 (HTTP {r.status_code}): {preview}")
                    print("[INFO] 立即停止重试。3 种最快恢复方式:")
                    print("  1) 去 https://platform.deepseek.com/user/wallet 充值 (秒开)")
                    print("  2) 临时换 Moonshot 新号: .env 中切换到 moonshot 预设 (免费额度)")
                    print("  3) 临时换 硅基流动/智谱/其他 OpenAI 兼容 API: .env 改 4 行")
                    break
                if _is_auth_error(r.status_code, data):
                    print(f"[FATAL] 鉴权失败 (HTTP {r.status_code}): {preview}")
                    print("[INFO] 立即停止重试。请核对:")
                    print("  - DEEPSEEK_API_KEY 是否从对应厂商控制台重新复制?")
                    print("  - 是否把 A 平台的 Key 填到了 B 厂商 (例如用 DeepSeek key 去请求 api.moonshot.cn)?")
                    print("  - Key 是否已在控制台被禁用/删除?")
                    break
                if r.status_code == 404:
                    err_obj = data.get("error") if isinstance(data, dict) else None
                    if isinstance(err_obj, dict):
                        err_msg = str(err_obj.get("message") or "")
                    elif isinstance(err_obj, str):
                        err_msg = err_obj
                    else:
                        err_msg = ""
                    print(f"[FATAL] HTTP 404 (模型或路径不存在): {preview}")
                    print("[INFO] 404 常见原因 + 修复:")
                    _dom = ""
                    if "://" in url:
                        _dom = url.split("://", 1)[1].split("/", 1)[0].rstrip(".")
                    if _dom == "api.moonshot.cn":
                        print(f"  → 你正在请求 Moonshot: 新号默认只开 kimi-k2 系列。当前 LLM_MODEL={LLM_MODEL!r}")
                        print("     建议改 .env: LLM_MODEL=kimi-k2.6   (或 kimi-k2.7-code)")
                        if "moonshot-v1" in LLM_MODEL:
                            print("     (moonshot-v1-8k / v1-128k / v1-auto 是旧版产品线，"
                                  "2026 新注册账号通常不包含；在 Model Arena 里能看到 kimi-k2.6 就说明是新账号)")
                    elif _dom.endswith("deepseek.com"):
                        print("  → 你正在请求 DeepSeek: 默认模型 deepseek-chat")
                    elif _dom == "open.bigmodel.cn":
                        print("  → 你正在请求 智谱 BigModel: 默认 glm-4-flash / glm-4-plus / glm-4.6")
                    elif _dom == "api.siliconflow.cn":
                        print("  → 你正在请求 硅基流动 SiliconFlow: 模型名要写厂商格式，如 deepseek-ai/DeepSeek-V2.5 / Qwen/Qwen2.5-72B-Instruct")
                    print("  → 如果都对不上，就是 LLM_BASE_URL 拼错了 (注意结尾必须是 /v1)")
                    print("  → 当前 .env 配置 (复制修改即可):")
                    print(f"       LLM_PROVIDER={LLM_PROVIDER!s}")
                    print(f"       LLM_BASE_URL={LLM_BASE_URL!s}")
                    print(f"       LLM_MODEL={LLM_MODEL!s}")
                    if err_msg:
                        print(f"  → 服务器返回: {err_msg}")
                    break

                print(f"[WARN] 第 {attempt} 次调用失败: HTTP {r.status_code}  {preview}")
                retriable_http = r.status_code in retry_on_http
                terminal_client = (r.status_code in terminal_client_http) and not retriable_http
                if terminal_client:
                    print(f"[INFO] 非重试类 HTTP {r.status_code}，停止重试。"
                          "404 请检查 LLM_BASE_URL/LLM_MODEL；422 说明请求体 schema 与厂商不兼容")
                    break

            if attempt < API_MAX_RETRIES:
                sleep_time = API_RETRY_DELAY * (2 ** (attempt - 1))
                print(f"[INFO] 等待 {sleep_time} 秒后重试...")
                time.sleep(sleep_time)
            continue

        except transient_exc as e:
            last_error = e
            print(f"[WARN] 第 {attempt} 次调用失败 (网络/超时类): {type(e).__name__}: {e}")
            if attempt < API_MAX_RETRIES:
                sleep_time = API_RETRY_DELAY * (2 ** (attempt - 1))
                print(f"[INFO] 等待 {sleep_time} 秒后重试...")
                time.sleep(sleep_time)
        except Exception as e:
            last_error = e
            print(f"[WARN] 第 {attempt} 次调用失败: {type(e).__name__}: {e}")
            if attempt < API_MAX_RETRIES:
                sleep_time = API_RETRY_DELAY * (2 ** (attempt - 1))
                print(f"[INFO] 等待 {sleep_time} 秒后重试...")
                time.sleep(sleep_time)
            else:
                break

    raise RuntimeError(
        f"{LLM_PROVIDER.upper()} API 调用失败，已重试 {API_MAX_RETRIES} 次。"
        f"最后错误: {type(last_error).__name__}: {last_error}"
    )


try:
    answer = generate(query, reranked_chunks)
    print("\n===== 最终答案 =====")
    print(answer)
except RuntimeError as e:
    print(f"\n[ERROR] {e}")
    print("\n[快速诊断]")
    print("  - 缺少 API Key → 在 .env 设置 DEEPSEEK_API_KEY=sk-... （https://platform.deepseek.com）")
    print("  - HTTP 401/403    → API Key 错了 / 没额度了")
    print("  - HTTP 404        → LLM_BASE_URL / LLM_MODEL 不对；默认用 deepseek-chat 别写别名")
    print("  - HTTP 429        → 速率/额度超限，稍后再试或升档")
    print("  - ConnectTimeout / SSL handshake timeout")
    print("                    → 看启动日志里的 LLM HTTP 层: NO PROXY SET 说明 .env 代理未启用；")
    print("                       '未见 Proxy 连接池' 说明代理配置没生效")
    print("  - 想换其他 OpenAI 兼容 API (Moonshot / 智谱 / 硅基流动 / Groq 等) 只要改 3 行：")
    print("       LLM_PROVIDER=...   LLM_BASE_URL=...   LLM_MODEL=...   DEEPSEEK_API_KEY=...")
    print("                       无需改代码")

