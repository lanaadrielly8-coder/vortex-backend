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

def get_groq_keys() -> list:
    """Retorna todas as keys Groq disponíveis."""
    keys = [k for k in [GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3,
                         os.getenv("GROQ_API_KEY_2",""), os.getenv("GROQ_API_KEY_3","")] if k]
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

CLAUDE_SONNET = "claude-sonnet-4-5"  # Elite
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

async def chamar_groq(
    messages: list,
    system: str = "",
    modelo: str = None,
    max_tokens: int = 2000,
) -> str:
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
    modelo: str = "gemini-2.0-flash",
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
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    
    if provedor_preferido in ["claude_sonnet", "claude_haiku"]:
        modelo_claude = CLAUDE_SONNET if provedor_preferido == "claude_sonnet" else CLAUDE_HAIKU
        # Elite/Pro: Claude primeiro, Gemini fallback — NUNCA Groq
        ordem = [
            ("claude", lambda: chamar_claude(messages, system, modelo_claude, max_tokens)),
            ("gemini", lambda: chamar_gemini(messages, system, "gemini-2.0-flash", max_tokens)),
        ]
    elif provedor_preferido == "gemini":
        ordem = [
            ("gemini", lambda: chamar_gemini(messages, system, "gemini-2.0-flash", max_tokens)),
            ("claude", lambda: chamar_claude(messages, system, CLAUDE_HAIKU, max_tokens)),
        ]
    else:
        # Free: Groq → Gemini (sem Claude)
        ordem = [
            ("groq", lambda: chamar_groq(messages, system, GROQ_MODEL_PADRAO, max_tokens)),
            ("gemini", lambda: chamar_gemini(messages, system, "gemini-2.0-flash", max_tokens)),
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
    
    # Fallback final — só Gemini para Free
    if not any("gemini" in e for e in erros):
        try:
            print("[VORTEX] 🆘 Fallback Gemini")
            texto = await chamar_gemini(messages, system, "gemini-2.0-flash", min(max_tokens, 1500))
            return texto, "gemini_fallback"
        except Exception as e:
            erros.append(f"gemini_last: {str(e)[:60]}")
    
    # Free sem mais opções — sugere upgrade
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
    voz_id: str = "pNInz6obpgDQGcFmaJgB",
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

# Aliases de compatibilidade
gerar_imagem_wavespeed = gerar_imagem_leonardo
gerar_video_wavespeed  = gerar_video_runway
