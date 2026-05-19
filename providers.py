"""
VORTEX AI — Providers v6.0 🔥
Temperatura 0.9 — Criatividade máxima
Llama 3.3 70B fixo — Qualidade Hollywood
"""

import os
import asyncio
import httpx
from fastapi import HTTPException

# ── URLs base ────────────────────────────────────────────────
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_BASE    = "https://generativelanguage.googleapis.com/v1beta/models"
LEONARDO_URL   = "https://cloud.leonardo.ai/api/rest/v1"
RUNWAY_URL     = "https://api.runwayml.com/v1"
ELEVENLABS_URL = "https://api.elevenlabs.io/v1"

# ── Chaves ───────────────────────────────────────────────────
# Suporte a múltiplas keys Groq em rotação
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY_2        = os.getenv("GROQ_API_KEY_2", "")
GROQ_API_KEY_3        = os.getenv("GROQ_API_KEY_3", "")
GROQ_API_KEY_4        = os.getenv("GROQ_API_KEY_4", "")
GROQ_API_KEY_5        = os.getenv("GROQ_API_KEY_5", "")
GROQ_API_KEY_6        = os.getenv("GROQ_API_KEY_6", "")

def get_groq_keys() -> list:
    """Retorna todas as keys Groq disponíveis."""
    keys = [k for k in [
        GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3,
        GROQ_API_KEY_4, GROQ_API_KEY_5, GROQ_API_KEY_6
    ] if k]
    return list(dict.fromkeys(keys))  # remove duplicatas

_groq_key_index = 0

def get_next_groq_key() -> str:
    """Rotaciona entre as keys disponíveis."""
    global _groq_key_index
    keys = get_groq_keys()
    if not keys: return ""
    key = keys[_groq_key_index % len(keys)]
    _groq_key_index += 1
    return key
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY", "")

CLAUDE_SONNET = "claude-sonnet-4-5-20251001"  # Elite
CLAUDE_HAIKU  = "claude-haiku-4-5-20251001"   # Pro
LEONARDO_API_KEY      = os.getenv("LEONARDO_API_KEY", "")
RUNWAY_API_KEY        = os.getenv("RUNWAY_API_KEY", "")
ELEVENLABS_API_KEY    = os.getenv("ELEVENLABS_API_KEY", "")
RAPIDAPI_KEY          = os.getenv("RAPIDAPI_KEY", "")
YOUTUBE_KEY           = os.getenv("YOUTUBE_KEY", "")
SHOTSTACK_SANDBOX_KEY = os.getenv("SHOTSTACK_SANDBOX_KEY", "")
SHOTSTACK_PROD_KEY    = os.getenv("SHOTSTACK_PROD_KEY", "")

# ── Modelo padrão — Llama 3.3 70B sempre ────────────────────
GROQ_MODEL_PADRAO = "llama-3.3-70b-versatile"

# ══════════════════════════════════════════════════════════════
# CONNECTION POOLS
# ══════════════════════════════════════════════════════════════

_groq_client: httpx.AsyncClient | None = None
_gemini_client: httpx.AsyncClient | None = None
_leonardo_client: httpx.AsyncClient | None = None
_runway_client: httpx.AsyncClient | None = None
_elevenlabs_client: httpx.AsyncClient | None = None
_rapidapi_client: httpx.AsyncClient | None = None
_youtube_client: httpx.AsyncClient | None = None


def get_groq_client() -> httpx.AsyncClient:
    global _groq_client
    if _groq_client is None or _groq_client.is_closed:
        _groq_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        )
    return _groq_client


def get_gemini_client() -> httpx.AsyncClient:
    global _gemini_client
    if _gemini_client is None or _gemini_client.is_closed:
        _gemini_client = httpx.AsyncClient(timeout=httpx.Timeout(35.0))
    return _gemini_client


def get_elevenlabs_client() -> httpx.AsyncClient:
    global _elevenlabs_client
    if _elevenlabs_client is None or _elevenlabs_client.is_closed:
        _elevenlabs_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        )
    return _elevenlabs_client


def get_rapidapi_client() -> httpx.AsyncClient:
    global _rapidapi_client
    if _rapidapi_client is None or _rapidapi_client.is_closed:
        _rapidapi_client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            headers={"x-rapidapi-key": RAPIDAPI_KEY},
        )
    return _rapidapi_client


def get_youtube_client() -> httpx.AsyncClient:
    global _youtube_client
    if _youtube_client is None or _youtube_client.is_closed:
        _youtube_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
    return _youtube_client


# ══════════════════════════════════════════════════════════════
# TEXTO — Groq (TEMPERATURA 0.9 — CRIATIVIDADE MÁXIMA)
# ══════════════════════════════════════════════════════════════

# Modelos Groq com rate limits independentes
GROQ_MODELOS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant", 
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]
_groq_modelo_idx = 0

async def chamar_groq(
    messages: list,
    system: str = "",
    modelo: str = None,
    max_tokens: int = 2000,
) -> str:
    global _groq_modelo_idx
    keys = get_groq_keys()
    if not keys:
        raise ValueError("GROQ_API_KEY não configurada")

    modelo_final = modelo or GROQ_MODEL_PADRAO
    if modelo_final == "llama-3.1-8b-instant":
        modelo_final = GROQ_MODEL_PADRAO

    payload = {
        "model": modelo_final,
        "max_tokens": max_tokens,
        "temperature": 0.9,
        "top_p": 0.95,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.2,
        "messages": ([{"role": "system", "content": system}] if system else []) + messages,
    }

    # Tenta cada key disponível
    last_error = None
    for i, key in enumerate(keys):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            ) as client:
                r = await client.post(GROQ_URL, json=payload)
                if r.status_code == 429:
                    print(f"[Groq] Key {i+1}/{len(keys)} bloqueada — tentando próxima...")
                    await asyncio.sleep(1)
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            if "429" in str(e) or "rate" in str(e).lower():
                print(f"[Groq] Key {i+1} rate limit — próxima key...")
                continue
            raise e
    
    raise ValueError(f"Todas as keys Groq bloqueadas: {last_error}")


# ══════════════════════════════════════════════════════════════
# TEXTO — Gemini (fallback)
# ══════════════════════════════════════════════════════════════

async def chamar_gemini(
    messages: list,
    system: str = "",
    modelo: str = "gemini-1.5-flash",
    max_tokens: int = 2000,
) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY não configurada")

    contents = []
    if system:
        contents.append({"role": "user", "parts": [{"text": f"[Sistema]: {system}"}]})
        contents.append({"role": "model", "parts": [{"text": "Entendido. Vou seguir essas instruções."}]})

    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.9,
            "topP": 0.95,
        }
    }

    client = get_gemini_client()
    r = await client.post(f"{GEMINI_BASE}/{modelo}:generateContent?key={GEMINI_API_KEY}", json=payload)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


# ══════════════════════════════════════════════════════════════
# TEXTO — Claude (Anthropic)
# ══════════════════════════════════════════════════════════════

async def chamar_claude(
    messages: list,
    system: str = "",
    modelo: str = None,
    max_tokens: int = 3000,
) -> str:
    key = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    if not key:
        raise ValueError("ANTHROPIC_API_KEY não configurada")
    
    modelo_final = modelo or CLAUDE_SONNET
    
    # Retry automático — Claude não pode falhar
    for tentativa in range(3):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": modelo_final,
                        "max_tokens": max_tokens,
                        "system": system,
                        "messages": messages,
                    },
                )
                if r.status_code == 529 or r.status_code == 503:
                    # Overloaded — espera e tenta de novo
                    wait = (tentativa + 1) * 3
                    print(f"[Claude] Sobrecarga — aguardando {wait}s (tentativa {tentativa+1}/3)...")
                    await asyncio.sleep(wait)
                    continue
                if r.status_code == 429:
                    # Rate limit — espera mais
                    wait = (tentativa + 1) * 5
                    print(f"[Claude] Rate limit — aguardando {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                if not r.is_success:
                    raise ValueError(f"Claude erro {r.status_code}: {r.text[:200]}")
                return r.json()["content"][0]["text"]
        except httpx.TimeoutException:
            if tentativa < 2:
                print(f"[Claude] Timeout — tentativa {tentativa+2}/3...")
                await asyncio.sleep(2)
                continue
            raise ValueError("Claude timeout após 3 tentativas")
    
    raise ValueError("Claude falhou após 3 tentativas")


# ══════════════════════════════════════════════════════════════
# TEXTO — Cascata Groq → Gemini
# ══════════════════════════════════════════════════════════════

async def gerar_texto(
    messages: list,
    system: str = "",
    max_tokens: int = 2000,
    provedor_preferido: str = "groq",
) -> tuple[str, str]:
    # Sempre começa com Groq 70b — mais estável e rápido
    # Claude e Gemini como fallback quando disponíveis
    
    if provedor_preferido in ["claude_sonnet", "claude_haiku"]:
        modelo_claude = CLAUDE_SONNET if provedor_preferido == "claude_sonnet" else CLAUDE_HAIKU
        ordem = [
            ("groq",        lambda: chamar_groq(messages, system, "llama-3.3-70b-versatile", max_tokens)),
            ("or_deepseek", lambda: chamar_openrouter(messages, system, "deepseek/deepseek-chat-v3-0324:free", max_tokens)),
            ("or_llama",    lambda: chamar_openrouter(messages, system, "meta-llama/llama-3.3-70b-instruct:free", max_tokens)),
            ("claude",      lambda: chamar_claude(messages, system, modelo_claude, max_tokens)),
            ("or_gemma",    lambda: chamar_openrouter(messages, system, "google/gemma-3-27b-it:free", max_tokens)),
            ("gemini",      lambda: chamar_gemini(messages, system, "gemini-1.5-flash", max_tokens)),
        ]
    elif provedor_preferido == "gemini":
        ordem = [
            ("groq",        lambda: chamar_groq(messages, system, "llama-3.3-70b-versatile", max_tokens)),
            ("or_deepseek", lambda: chamar_openrouter(messages, system, "deepseek/deepseek-chat-v3-0324:free", max_tokens)),
            ("or_llama",    lambda: chamar_openrouter(messages, system, "meta-llama/llama-3.3-70b-instruct:free", max_tokens)),
            ("gemini",      lambda: chamar_gemini(messages, system, "gemini-1.5-flash", max_tokens)),
            ("or_gemma",    lambda: chamar_openrouter(messages, system, "google/gemma-3-27b-it:free", max_tokens)),
            ("claude",      lambda: chamar_claude(messages, system, CLAUDE_HAIKU, max_tokens)),
        ]
    else:
        # Cascata COMPLETA — Groq x6 + 39 modelos OpenRouter + Gemini + Claude
        # NUNCA cai — 42 IAs em sequência
        ordem = [
            ("groq",          lambda: chamar_groq(messages, system, "llama-3.3-70b-versatile", max_tokens)),
            ("or_deepseek_v3", lambda m="deepseek/deepseek-chat-v3-0324:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_deepseek_r1", lambda m="deepseek/deepseek-r1:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_deepseek_r1_zero", lambda m="deepseek/deepseek-r1-zero:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_llama33_70b", lambda m="meta-llama/llama-3.3-70b-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_llama31_8b", lambda m="meta-llama/llama-3.1-8b-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_llama31_405b", lambda m="meta-llama/llama-3.1-405b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_llama32_11b", lambda m="meta-llama/llama-3.2-11b-vision-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_llama32_3b", lambda m="meta-llama/llama-3.2-3b-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_llama32_1b", lambda m="meta-llama/llama-3.2-1b-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_qwen3_235b", lambda m="qwen/qwen3-235b-a22b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_qwen3_30b", lambda m="qwen/qwen3-30b-a3b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_qwen3_14b", lambda m="qwen/qwen3-14b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_qwen3_8b", lambda m="qwen/qwen3-8b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_qwen25_72b", lambda m="qwen/qwen-2.5-72b-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_qwen25_7b", lambda m="qwen/qwen-2.5-7b-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_qwq_32b", lambda m="qwen/qwq-32b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_gemma3_27b", lambda m="google/gemma-3-27b-it:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_gemma3_12b", lambda m="google/gemma-3-12b-it:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_gemma3_4b", lambda m="google/gemma-3-4b-it:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_gemma3_1b", lambda m="google/gemma-3-1b-it:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_gemma2_9b", lambda m="google/gemma-2-9b-it:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_phi4", lambda m="microsoft/phi-4:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_phi4_mini", lambda m="microsoft/phi-4-mini-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_phi3_mini", lambda m="microsoft/phi-3-mini-128k-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_phi3_medium", lambda m="microsoft/phi-3-medium-128k-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_mistral_7b", lambda m="mistralai/mistral-7b-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_mistral_nemo", lambda m="mistralai/mistral-nemo:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_hermes3_405b", lambda m="nousresearch/hermes-3-llama-3.1-405b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_hermes3_70b", lambda m="nousresearch/hermes-3-llama-3.1-70b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_dolphin_r1", lambda m="cognitivecomputations/dolphin3.0-r1-mistral-nemo:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_dolphin_llama", lambda m="cognitivecomputations/dolphin-mixtral-8x22b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_toppy_7b", lambda m="undi95/toppy-m-7b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_mythomax", lambda m="gryphe/mythomax-l2-13b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_openchat_7b", lambda m="openchat/openchat-7b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_zephyr_7b", lambda m="huggingfaceh4/zephyr-7b-beta:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_yi_34b", lambda m="01-ai/yi-34b-chat:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_solar_10b", lambda m="upstage/solar-10.7b-instruct:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_gemini_flash", lambda m="google/gemini-flash-1.5-8b:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("or_gemini_flash_exp", lambda m="google/gemini-flash-1.5:free": chamar_openrouter(messages, system, m, max_tokens)),
            ("gemini",        lambda: chamar_gemini(messages, system, "gemini-1.5-flash", max_tokens)),
            ("claude",        lambda: chamar_claude(messages, system, CLAUDE_HAIKU, max_tokens)),
        ]
    erros = []
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    
    # Cascata silenciosa: tenta cada provider, se falhar passa pro próximo
    for nome, chamada in ordem:
        try:
            texto = await chamada()
            print(f"[VORTEX] ✅ {nome} — {len(texto)} chars")
            return texto, nome
        except Exception as e:
            erro_str = str(e).lower()
            erros.append(f"{nome}: {str(e)[:60]}")
            print(f"[VORTEX] ⚠️ {nome} falhou — tentando próximo...")
            
            # Rate limit no Groq — espera 2s e continua
            if any(x in erro_str for x in ["rate","429","limit","quota","too many"]):
                await asyncio.sleep(2)
            continue
    
    # Fallback final — tenta OpenRouter com modelo menor
    try:
        print("[VORTEX] 🆘 Fallback final → OpenRouter")
        texto = await chamar_openrouter(messages, system, "mistralai/mistral-7b-instruct:free", min(max_tokens, 1500))
        return texto, "openrouter_fallback"
    except Exception as e:
        erros.append(f"openrouter_last: {str(e)[:60]}")
    try:
        print("[VORTEX] 🆘 Último recurso → Groq 8b")
        texto = await chamar_groq(messages, system, "llama-3.1-8b-instant", min(max_tokens, 1000))
        return texto, "groq_fallback"
    except Exception as e:
        erros.append(f"groq_last: {str(e)[:60]}")

    raise HTTPException(503, "Serviço ocupado. Tente novamente em 1 minuto. Planos Pro e Elite têm prioridade garantida.")


# ══════════════════════════════════════════════════════════════
# IMAGEM — Leonardo AI
# ══════════════════════════════════════════════════════════════

async def gerar_imagem_leonardo(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    modelo: str = "PHOENIX",
) -> str:
    if not LEONARDO_API_KEY:
        raise HTTPException(500, "LEONARDO_API_KEY não configurada")

    MODELOS_ID = {
        "PHOENIX": "6bef9f1b-29cb-40c7-b9df-32b51c1f67d3",
        "LEONARDO_CREATIVE": "b24e16ff-06e3-43eb-8d33-4416c2d75876",
        "LEONARDO_SIGNATURE": "291be633-cb24-434f-898f-e662799936ad",
    }
    modelo_id = MODELOS_ID.get(modelo, MODELOS_ID["PHOENIX"])

    payload = {
        "prompt": prompt,
        "modelId": modelo_id,
        "width": width,
        "height": height,
        "num_images": 1,
        "negative_prompt": "blurry, low quality, distorted, ugly, deformed",
        "guidance_scale": 7,
        "num_inference_steps": 30,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0),
        headers={"Authorization": f"Bearer {LEONARDO_API_KEY}", "Content-Type": "application/json"}) as client:
        r = await client.post(f"{LEONARDO_URL}/generations", json=payload)
        if r.status_code != 200:
            raise HTTPException(502, f"Leonardo erro {r.status_code}: {r.text[:200]}")
        data = r.json()
        generation_id = data.get("sdGenerationJob", {}).get("generationId")
        if not generation_id:
            raise HTTPException(502, "Leonardo não retornou generation_id")

        elapsed = 0
        while elapsed < 120:
            await asyncio.sleep(3)
            elapsed += 3
            r = await client.get(f"{LEONARDO_URL}/generations/{generation_id}")
            data = r.json()
            generations = data.get("generations_by_pk", {}).get("generated_images", [])
            if generations:
                url = generations[0].get("url")
                if url: return url
            if data.get("generations_by_pk", {}).get("status") == "FAILED":
                raise HTTPException(502, "Leonardo geração falhou")

    raise HTTPException(504, "Timeout Leonardo")


# ══════════════════════════════════════════════════════════════
# VÍDEO — Runway Gen-3
# ══════════════════════════════════════════════════════════════

async def gerar_video_runway(
    prompt: str,
    duracao: int = 5,
    resolucao: str = "720p",
    ratio: str = "16:9",
) -> str:
    if not RUNWAY_API_KEY:
        raise HTTPException(500, "RUNWAY_API_KEY não configurada")

    payload = {"promptText": prompt, "model": "gen3a_turbo", "duration": duracao, "ratio": ratio, "watermark": False}

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0),
        headers={"Authorization": f"Bearer {RUNWAY_API_KEY}", "Content-Type": "application/json"}) as client:
        r = await client.post(f"{RUNWAY_URL}/tasks", json=payload)
        if r.status_code != 200:
            raise HTTPException(502, f"Runway erro {r.status_code}: {r.text[:200]}")
        task_id = r.json().get("id")
        if not task_id:
            raise HTTPException(502, "Runway sem task_id")

        elapsed = 0
        while elapsed < 360:
            await asyncio.sleep(10)
            elapsed += 10
            r = await client.get(f"{RUNWAY_URL}/tasks/{task_id}")
            data = r.json()
            status = data.get("status")
            if status == "SUCCEEDED":
                artifacts = data.get("artifacts", [])
                if artifacts:
                    url = artifacts[0].get("url")
                    if url: return url
                raise HTTPException(502, "Runway sucesso sem URL")
            if status in ["FAILED", "CANCELED"]:
                raise HTTPException(502, f"Runway falhou: {data.get('error',{}).get('message','')}")
            print(f"[Runway] {status} ({elapsed}s)")

    raise HTTPException(504, "Timeout Runway")


# ══════════════════════════════════════════════════════════════
# VOZ — ElevenLabs
# ══════════════════════════════════════════════════════════════

async def gerar_voz_elevenlabs(
    texto: str,
    voz_id: str = "onwK4e9ZLuTAKqWW03F9",  # VORTEX oficial
    modelo: str = "eleven_multilingual_v2",
) -> str:
    if not ELEVENLABS_API_KEY:
        raise HTTPException(500, "ELEVENLABS_API_KEY não configurada")

    payload = {
        "text": texto,
        "model_id": modelo,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.3, "use_speaker_boost": True},
    }

    client = get_elevenlabs_client()
    r = await client.post(f"{ELEVENLABS_URL}/text-to-speech/{voz_id}", json=payload)

    if r.status_code != 200:
        raise HTTPException(502, f"ElevenLabs erro {r.status_code}: {r.text[:200]}")

    import base64
    audio_b64 = base64.b64encode(r.content).decode("utf-8")
    return f"data:audio/mpeg;base64,{audio_b64}"


# ══════════════════════════════════════════════════════════════
# ANÁLISE — Instagram
# ══════════════════════════════════════════════════════════════

async def analisar_instagram(perfil: str) -> dict:
    key = os.getenv("RAPIDAPI_KEY", RAPIDAPI_KEY)
    if not key:
        raise HTTPException(500, "RAPIDAPI_KEY não configurada")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        r = await client.get(
            "https://instagram-scraper-20251.p.rapidapi.com/userinfo/",
            params={"username": perfil},
            headers={
                "x-rapidapi-key": key,
                "x-rapidapi-host": "instagram-scraper-20251.p.rapidapi.com"
            }
        )
        if not r.is_success:
            raise HTTPException(502, f"Instagram API erro {r.status_code}: {r.text[:200]}")
        
        data = r.json()
        user = data.get("data", data)
        
        seg = user.get("follower_count", user.get("followers", 0))
        seg_num = int(seg) if str(seg).isdigit() else 0
        posts = user.get("media_count", user.get("posts_count", 0))
        eng = user.get("engagement_rate", "")
        
        return {
            "seguidores": formatar_numero(seg_num),
            "seguindo": formatar_numero(int(user.get("following_count", user.get("following", 0)) or 0)),
            "posts": str(posts),
            "engajamento": f"{eng}%" if eng else "N/A",
            "verificado": user.get("is_verified", False),
            "bio": (user.get("biography", user.get("bio", "")) or "")[:150],
            "nome": user.get("full_name", user.get("name", perfil)),
            "avatar": user.get("profile_pic_url", ""),
        }


# ══════════════════════════════════════════════════════════════
# ANÁLISE — TikTok
# ══════════════════════════════════════════════════════════════

async def analisar_tiktok(perfil: str) -> dict:
    key = os.getenv("RAPIDAPI_KEY", RAPIDAPI_KEY)
    if not key:
        raise HTTPException(500, "RAPIDAPI_KEY não configurada")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        r = await client.get(
            "https://tiktok-scraper7.p.rapidapi.com/user/info",
            params={"unique_id": perfil},
            headers={
                "x-rapidapi-key": key,
                "x-rapidapi-host": "tiktok-scraper7.p.rapidapi.com"
            }
        )
        if not r.is_success:
            # Tenta endpoint TIKWM como fallback
            r2 = await client.get(
                "https://tiktok-scraper-2022.p.rapidapi.com/user/info",
                params={"user_id": perfil},
                headers={
                    "x-rapidapi-key": key,
                    "x-rapidapi-host": "tiktok-scraper-2022.p.rapidapi.com"
                }
            )
            if not r2.is_success:
                raise HTTPException(502, f"TikTok API erro {r.status_code}")
            r = r2
        
        data = r.json()
        user_info = data.get("data", {}).get("user", data.get("data", {}))
        stats = user_info.get("stats", data.get("data", {}).get("stats", {}))
        
        seg = stats.get("followerCount", stats.get("follower_count", 0))
        cur = stats.get("heartCount", stats.get("digg_count", 0))
        seg_num = int(seg) if seg else 0
        cur_num = int(cur) if cur else 0
        
        return {
            "seguidores": formatar_numero(seg_num),
            "curtidas": formatar_numero(cur_num),
            "videos": str(stats.get("videoCount", stats.get("video_count", 0))),
            "engajamento": f"{round((cur_num/max(seg_num,1))*100,1)}%" if seg_num else "N/A",
            "verificado": user_info.get("verified", False),
            "nome": user_info.get("nickname", perfil),
            "bio": (user_info.get("signature", "") or "")[:150],
            "avatar": user_info.get("avatarMedium", user_info.get("avatar_medium", {}).get("url_list", [""])[0] if isinstance(user_info.get("avatar_medium"), dict) else ""),
        }


# ══════════════════════════════════════════════════════════════
# ANÁLISE — YouTube
# ══════════════════════════════════════════════════════════════

async def analisar_youtube(perfil: str) -> dict:
    if not YOUTUBE_KEY:
        raise HTTPException(500, "YOUTUBE_KEY não configurada")
    client = get_youtube_client()
    r = await client.get("https://www.googleapis.com/youtube/v3/search",
        params={"part":"snippet","q":perfil,"type":"channel","key":YOUTUBE_KEY})
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items: raise HTTPException(404, "Canal não encontrado")
    channel_id = items[0]["id"]["channelId"]
    r = await client.get("https://www.googleapis.com/youtube/v3/channels",
        params={"part":"statistics,snippet","id":channel_id,"key":YOUTUBE_KEY})
    r.raise_for_status()
    canal = r.json()["items"][0]
    stats = canal["statistics"]
    snippet = canal["snippet"]
    return {
        "seguidores": formatar_numero(int(stats.get("subscriberCount", 0))),
        "views_total": formatar_numero(int(stats.get("viewCount", 0))),
        "videos": str(stats.get("videoCount", 0)),
        "engajamento": "Dados ao vivo",
        "descricao": (snippet.get("description", "") or "")[:120],
    }


# ══════════════════════════════════════════════════════════════
# UTILITÁRIO
# ══════════════════════════════════════════════════════════════

def formatar_numero(n: int) -> str:
    if n >= 1_000_000: return f"{round(n/1_000_000,1)}M"
    elif n >= 1_000: return f"{round(n/1_000,1)}K"
    return str(n)


# ══════════════════════════════════════════════════════════════
# VÍDEO — Kling AI (melhor custo-benefício)
# ══════════════════════════════════════════════════════════════

async def gerar_video_kling(
    prompt: str,
    duracao: int = 5,
    ratio: str = "9:16",
    modelo: str = "kling-v1",
    imagem_base64: str = None,
) -> str:
    """Gera vídeo com Kling AI via RapidAPI."""
    key = os.getenv("RAPIDAPI_KEY", RAPIDAPI_KEY)
    if not key:
        raise HTTPException(500, "RAPIDAPI_KEY não configurada")

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        # Submete o job
        r = await client.post(
            "https://kling-ai1.p.rapidapi.com/v1/videos/text2video",
            headers={
                "x-rapidapi-key": key,
                "x-rapidapi-host": "kling-ai1.p.rapidapi.com",
                "Content-Type": "application/json",
            },
            json={
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted",
                "cfg_scale": 0.5,
                "mode": "std",
                "aspect_ratio": ratio,
                "duration": str(duracao),
                "model_name": modelo,
            },
        )
        if not r.is_success:
            raise HTTPException(502, f"Kling erro {r.status_code}: {r.text[:200]}")
        
        data = r.json()
        task_id = data.get("data", {}).get("task_id")
        if not task_id:
            raise HTTPException(502, f"Kling sem task_id: {data}")
        
        print(f"[Kling] task_id: {task_id}")
        
        # Polling
        elapsed = 0
        while elapsed < 300:
            await asyncio.sleep(8)
            elapsed += 8
            
            r2 = await client.get(
                f"https://api.klingai.com/v1/videos/text2video/{task_id}",
                headers={"Authorization": f"Bearer {kling_jwt()}"},
            )
            if r2.is_success:
                d2 = r2.json()
                status = d2.get("data", {}).get("task_status")
                print(f"[Kling] status={status} ({elapsed}s)")
                
                if status == "succeed":
                    works = d2.get("data", {}).get("task_result", {}).get("videos", [])
                    if works:
                        url = works[0].get("url")
                        if url: return url
                    raise HTTPException(502, "Kling sucesso mas sem URL")
                
                if status == "failed":
                    raise HTTPException(502, f"Kling falhou: {d2}")
        
        raise HTTPException(504, "Kling timeout (5min)")


# ══════════════════════════════════════════════════════════════
# MÚSICA — ElevenLabs Sound Effects
# ══════════════════════════════════════════════════════════════

async def gerar_musica_elevenlabs(
    prompt: str,
    duracao: int = 30,
) -> str:
    """Gera música via ElevenLabs — requer plano Creator ou superior."""
    key = os.getenv("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)
    if not key:
        raise HTTPException(500, "ELEVENLABS_API_KEY não configurada")

    # Sound generation requer plano pago — usar endpoint correto
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(
            "https://api.elevenlabs.io/v1/sound-generation",
            headers={
                "xi-api-key": key,
                "Content-Type": "application/json",
            },
            json={
                "text": prompt,
                "duration_seconds": min(duracao, 22),  # max 22s na API
                "prompt_influence": 0.3,
            },
        )
        if not r.is_success:
            raise HTTPException(502, f"ElevenLabs Sound erro {r.status_code}: {r.text[:200]}")
        
        import base64
        audio_b64 = base64.b64encode(r.content).decode("utf-8")
        return f"data:audio/mpeg;base64,{audio_b64}"


# ══════════════════════════════════════════════════════════════
# VÍDEO — Luma Dream Machine
# ══════════════════════════════════════════════════════════════
async def gerar_video_luma(prompt: str, duracao: int = 5, modelo: str = "dream-machine") -> str:
    key = os.getenv("LUMAAI_API_KEY", "")
    if not key:
        raise HTTPException(500, "LUMAAI_API_KEY não configurada — pegar em lumalabs.ai")
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        r = await client.post(
            "https://api.lumalabs.ai/dream-machine/v1/generations",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"prompt": prompt, "model": modelo, "duration": min(duracao, 10), "resolution": "720p", "aspect_ratio": "9:16"},
        )
        if not r.is_success:
            raise HTTPException(502, f"Luma erro {r.status_code}: {r.text[:200]}")
        task_id = r.json().get("id")
        if not task_id:
            raise HTTPException(502, "Luma sem task_id")
        elapsed = 0
        while elapsed < 300:
            await asyncio.sleep(8); elapsed += 8
            r2 = await client.get(
                f"https://api.lumalabs.ai/dream-machine/v1/generations/{task_id}",
                headers={"Authorization": f"Bearer {key}"},
            )
            if r2.is_success:
                d = r2.json()
                state = d.get("state")
                print(f"[Luma] state={state} ({elapsed}s)")
                if state == "completed":
                    url = d.get("assets", {}).get("video")
                    if url: return url
                    raise HTTPException(502, "Luma sem URL")
                if state == "failed":
                    raise HTTPException(502, f"Luma falhou: {d.get('failure_reason','')}")
        raise HTTPException(504, "Luma timeout")


# ══════════════════════════════════════════════════════════════
# VÍDEO — MiniMax Video-01
# ══════════════════════════════════════════════════════════════
async def gerar_video_minimax(prompt: str, duracao: int = 5) -> str:
    key = os.getenv("MINIMAX_API_KEY", "")
    if not key:
        raise HTTPException(500, "MINIMAX_API_KEY não configurada — pegar em minimaxi.com")
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        r = await client.post(
            "https://api.minimaxi.chat/v1/video_generation",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "video-01", "prompt": prompt},
        )
        if not r.is_success:
            raise HTTPException(502, f"MiniMax erro {r.status_code}: {r.text[:200]}")
        task_id = r.json().get("task_id")
        if not task_id:
            raise HTTPException(502, "MiniMax sem task_id")
        elapsed = 0
        while elapsed < 300:
            await asyncio.sleep(8); elapsed += 8
            r2 = await client.get(
                f"https://api.minimaxi.chat/v1/query/video_generation?task_id={task_id}",
                headers={"Authorization": f"Bearer {key}"},
            )
            if r2.is_success:
                d = r2.json()
                status = d.get("status")
                print(f"[MiniMax] status={status} ({elapsed}s)")
                if status == "Success":
                    url = d.get("file_id","")
                    if url: return f"https://api.minimaxi.chat/v1/files/retrieve?GroupId=&file_id={url}"
                    raise HTTPException(502, "MiniMax sem URL")
                if status in ["Failed","Fail"]:
                    raise HTTPException(502, "MiniMax falhou")
        raise HTTPException(504, "MiniMax timeout")


# ══════════════════════════════════════════════════════════════
# VÍDEO — Google Veo 2 (via Gemini API)
# ══════════════════════════════════════════════════════════════
async def gerar_video_veo(prompt: str, duracao: int = 5) -> str:
    import base64
    key = os.getenv("VEO_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not key:
        raise HTTPException(500, "VEO_API_KEY não configurada — use mesma key do Gemini")

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        # Submete geração
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/veo-2.0-generate-001:generateVideo?key={key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "durationSeconds": min(duracao, 8),
                    "aspectRatio": "9:16",
                    "personGeneration": "dont_allow",
                }
            },
        )
        if not r.is_success:
            raise HTTPException(502, f"Veo 2 erro {r.status_code}: {r.text[:300]}")

        data = r.json()
        op_name = data.get("name")
        if not op_name:
            raise HTTPException(502, f"Veo 2 sem operation name: {data}")

        print(f"[Veo 2] operation: {op_name}")

        # Polling da operação
        elapsed = 0
        while elapsed < 300:
            await asyncio.sleep(10); elapsed += 10
            r2 = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/{op_name}?key={key}",
            )
            if r2.is_success:
                d2 = r2.json()
                done = d2.get("done", False)
                print(f"[Veo 2] done={done} ({elapsed}s)")
                if done:
                    videos = d2.get("response", {}).get("videos", [])
                    if videos:
                        video_data = videos[0].get("bytesBase64Encoded", "")
                        if video_data:
                            import tempfile, uuid
                            fname = f"veo_{uuid.uuid4().hex}.mp4"
                            fpath = os.path.join("uploads", fname)
                            os.makedirs("uploads", exist_ok=True)
                            with open(fpath, "wb") as f:
                                f.write(base64.b64decode(video_data))
                            vortex_url = os.getenv("VORTEX_URL", "http://127.0.0.1:8082")
                            return f"{vortex_url}/uploads/{fname}"
                    raise HTTPException(502, "Veo 2 sem vídeo na resposta")
        raise HTTPException(504, "Veo 2 timeout (5min)")


# ══════════════════════════════════════════════════════════════
# IMAGEM — Stability AI (SDXL + SD3)
# ══════════════════════════════════════════════════════════════
async def gerar_imagem_stability(prompt: str, width: int = 1024, height: int = 1024, modelo: str = "stable-diffusion-xl-1024-v1-0") -> str:
    import base64
    key = os.getenv("STABILITY_API_KEY", "")
    if not key:
        raise HTTPException(500, "STABILITY_API_KEY não configurada — pegar em platform.stability.ai")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/ultra",
            headers={"authorization": f"Bearer {key}", "accept": "application/json"},
            data={"prompt": prompt, "output_format": "jpeg", "width": width, "height": height},
        )
        if not r.is_success:
            raise HTTPException(502, f"Stability erro {r.status_code}: {r.text[:200]}")
        d = r.json()
        img_b64 = d.get("image", "")
        if not img_b64:
            raise HTTPException(502, "Stability sem imagem")
        return f"data:image/jpeg;base64,{img_b64}"


# ══════════════════════════════════════════════════════════════
# IMAGEM — Ideogram v2 (melhor texto em imagem)
# ══════════════════════════════════════════════════════════════
async def gerar_imagem_ideogram(prompt: str, width: int = 1024, height: int = 1024) -> str:
    import base64
    key = os.getenv("IDEOGRAM_API_KEY", "")
    if not key:
        raise HTTPException(500, "IDEOGRAM_API_KEY não configurada — pegar em ideogram.ai/api")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(
            "https://api.ideogram.ai/generate",
            headers={"Api-Key": key, "Content-Type": "application/json"},
            json={
                "image_request": {
                    "prompt": prompt,
                    "model": "V_2",
                    "resolution": "RESOLUTION_1024_1024",
                    "style_type": "REALISTIC",
                }
            },
        )
        if not r.is_success:
            raise HTTPException(502, f"Ideogram erro {r.status_code}: {r.text[:200]}")
        d = r.json()
        url = d.get("data", [{}])[0].get("url", "")
        if not url:
            raise HTTPException(502, "Ideogram sem URL")
        return url


# ══════════════════════════════════════════════════════════════
# MÚSICA — Suno AI (música completa com letra)
# ══════════════════════════════════════════════════════════════
async def gerar_musica_suno(prompt: str, duracao: int = 30, instrumental: bool = True) -> str:
    key = os.getenv("SUNO_API_KEY", "")
    if not key:
        raise HTTPException(500, "SUNO_API_KEY não configurada — pegar em suno.com/api")
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        r = await client.post(
            "https://api.suno.ai/v1/generations",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "prompt": prompt,
                "instrumental": instrumental,
                "duration": min(duracao, 120),
            },
        )
        if not r.is_success:
            raise HTTPException(502, f"Suno erro {r.status_code}: {r.text[:200]}")
        d = r.json()
        task_id = d.get("id")
        if not task_id:
            raise HTTPException(502, "Suno sem task_id")
        elapsed = 0
        while elapsed < 180:
            await asyncio.sleep(8); elapsed += 8
            r2 = await client.get(
                f"https://api.suno.ai/v1/generations/{task_id}",
                headers={"Authorization": f"Bearer {key}"},
            )
            if r2.is_success:
                d2 = r2.json()
                status = d2.get("status")
                print(f"[Suno] status={status} ({elapsed}s)")
                if status == "complete":
                    url = d2.get("audio_url", "")
                    if url: return url
                    raise HTTPException(502, "Suno sem audio_url")
                if status == "failed":
                    raise HTTPException(502, "Suno falhou")
        raise HTTPException(504, "Suno timeout")


# ══════════════════════════════════════════════════════════════
# MÚSICA — Udio (música + voz cantada)
# ══════════════════════════════════════════════════════════════
async def gerar_musica_udio(prompt: str) -> str:
    key = os.getenv("UDIO_API_KEY", "")
    if not key:
        raise HTTPException(500, "UDIO_API_KEY não configurada — pegar em udio.com/api")
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        r = await client.post(
            "https://www.udio.com/api/generate-proxy",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"prompt": prompt, "samplerOptions": {"seed": -1}},
        )
        if not r.is_success:
            raise HTTPException(502, f"Udio erro {r.status_code}: {r.text[:200]}")
        d = r.json()
        tracks = d.get("tracks", [])
        if not tracks:
            raise HTTPException(502, "Udio sem tracks")
        return tracks[0].get("song_path", "")


# ══════════════════════════════════════════════════════════════
# TRANSCRIÇÃO — AssemblyAI
# ══════════════════════════════════════════════════════════════
async def transcrever_assemblyai(audio_url: str) -> list:
    key = os.getenv("ASSEMBLYAI_API_KEY", "")
    if not key:
        raise HTTPException(500, "ASSEMBLYAI_API_KEY não configurada — pegar em assemblyai.com")
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        r = await client.post(
            "https://api.assemblyai.com/v2/transcript",
            headers={"authorization": key, "content-type": "application/json"},
            json={"audio_url": audio_url, "language_code": "pt", "word_boost": [], "format_text": True},
        )
        if not r.is_success:
            raise HTTPException(502, f"AssemblyAI erro {r.status_code}")
        tid = r.json().get("id")
        if not tid:
            raise HTTPException(502, "AssemblyAI sem ID")
        elapsed = 0
        while elapsed < 300:
            await asyncio.sleep(5); elapsed += 5
            r2 = await client.get(
                f"https://api.assemblyai.com/v2/transcript/{tid}",
                headers={"authorization": key},
            )
            if r2.is_success:
                d2 = r2.json()
                status = d2.get("status")
                print(f"[AssemblyAI] status={status} ({elapsed}s)")
                if status == "completed":
                    return d2.get("words", [])
                if status == "error":
                    raise HTTPException(502, f"AssemblyAI erro: {d2.get('error')}")
        raise HTTPException(504, "AssemblyAI timeout")


# ══════════════════════════════════════════════════════════════
# AVATAR — Hedra (rosto falando com áudio)
# ══════════════════════════════════════════════════════════════
async def gerar_avatar_hedra(imagem_url: str, audio_url: str) -> str:
    key = os.getenv("HEDRA_API_KEY", "")
    if not key:
        raise HTTPException(500, "HEDRA_API_KEY não configurada — pegar em hedra.com")
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        r = await client.post(
            "https://mercury.dev.dream-ai.com/api/v1/characters",
            headers={"X-API-Key": key, "Content-Type": "application/json"},
            json={"text": "", "voice_url": audio_url, "avatar_image_url": imagem_url, "aspect_ratio": "9:16"},
        )
        if not r.is_success:
            raise HTTPException(502, f"Hedra erro {r.status_code}: {r.text[:200]}")
        d = r.json()
        job_id = d.get("jobId") or d.get("job_id")
        if not job_id:
            raise HTTPException(502, "Hedra sem job_id")
        elapsed = 0
        while elapsed < 300:
            await asyncio.sleep(8); elapsed += 8
            r2 = await client.get(
                f"https://mercury.dev.dream-ai.com/api/v1/projects/{job_id}",
                headers={"X-API-Key": key},
            )
            if r2.is_success:
                d2 = r2.json()
                status = d2.get("status")
                print(f"[Hedra] status={status} ({elapsed}s)")
                if status == "Completed":
                    url = d2.get("videoUrl") or d2.get("video_url", "")
                    if url: return url
                    raise HTTPException(502, "Hedra sem videoUrl")
                if status in ["Failed", "Error"]:
                    raise HTTPException(502, "Hedra falhou")
        raise HTTPException(504, "Hedra timeout")

# Aliases de compatibilidade
gerar_imagem_wavespeed = gerar_imagem_leonardo
gerar_video_wavespeed  = gerar_video_runway
# ── MISTRAL ──────────────────────────────────────────────────────────────────
async def chamar_mistral(messages: list, system: str = "", modelo: str = "mistral-large-latest", max_tokens: int = 2000) -> str:
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY não configurada")
    msgs = ([{"role":"system","content":system}] if system else []) + messages
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={"model": modelo, "messages": msgs, "max_tokens": max_tokens}
        )
        if r.status_code != 200:
            raise ValueError(f"Mistral erro {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]

# ── COHERE ────────────────────────────────────────────────────────────────────
async def chamar_cohere(messages: list, system: str = "", modelo: str = "command-r-plus", max_tokens: int = 2000) -> str:
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY não configurada")
    # Cohere usa formato diferente
    chat_history = []
    for m in messages[:-1]:
        chat_history.append({"role": "USER" if m["role"]=="user" else "CHATBOT", "message": m["content"]})
    last_msg = messages[-1]["content"] if messages else ""
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(
            "https://api.cohere.com/v1/chat",
            headers={"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"},
            json={"model": modelo, "message": last_msg, "chat_history": chat_history, "preamble": system, "max_tokens": max_tokens}
        )
        if r.status_code != 200:
            raise ValueError(f"Cohere erro {r.status_code}: {r.text[:200]}")
        return r.json()["text"]

# ── TOGETHER AI ───────────────────────────────────────────────────────────────
async def chamar_together(messages: list, system: str = "", modelo: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo", max_tokens: int = 2000) -> str:
    if not TOGETHER_API_KEY:
        raise ValueError("TOGETHER_API_KEY não configurada")
    msgs = ([{"role":"system","content":system}] if system else []) + messages
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
            json={"model": modelo, "messages": msgs, "max_tokens": max_tokens}
        )
        if r.status_code != 200:
            raise ValueError(f"Together erro {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]

# ── OPENROUTER ────────────────────────────────────────────────────────────────
async def chamar_openrouter(messages: list, system: str = "", modelo: str = "deepseek/deepseek-chat", max_tokens: int = 2000) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY não configurada")
    msgs = ([{"role":"system","content":system}] if system else []) + messages
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": modelo, "messages": msgs, "max_tokens": max_tokens}
        )
        if r.status_code != 200:
            raise ValueError(f"OpenRouter erro {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]