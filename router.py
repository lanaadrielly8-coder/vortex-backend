"""
VORTEX AI — Router Inteligente v4.1
Correções:
  ✅ "roteiro curto" → SIMPLES (8B rápido, não 70B)
  ✅ "roteiro" sozinho → MEDIA (70B quando necessário)
  ✅ Keywords de onboarding classificadas como SIMPLES
  ✅ Tokens máximos alinhados com necessidade real
"""

import os
import re
from enum import Enum


class Complexidade(str, Enum):
    SIMPLES  = "simples"   # resposta rápida, onboarding, FAQ, roteiro curto
    MEDIA    = "media"     # análise, sugestões, roteiro médio
    COMPLEXA = "complexa"  # roteiro longo, estratégia, campanha, série


# ── Complexa — prioridade máxima ─────────────────────────────
_KEYWORDS_COMPLEXA = [
    "roteiro épico", "roteiro cinematográfico", "roteiro longo",
    "estratégia completa", "plano de conteúdo", "série de vídeos",
    "campanha", "storytelling profundo", "análise profunda",
    "detalhado", "extenso", "script completo", "screenplay",
    "documentário", "masterplan", "plano anual", "calendário editorial",
]

# ── Simples — resposta rápida ─────────────────────────────────
_KEYWORDS_SIMPLES = [
    # Perguntas básicas
    "resumo", "resuma", "o que é", "defina", "explique brevemente",
    "dica rápida", "exemplo", "liste", "quais são", "me diz",
    "significa", "diferença entre", "quanto custa", "quando",
    # Roteiros curtos — 8B é suficiente e MUITO mais rápido
    "roteiro curto", "roteiro rápido", "roteiro reels", "roteiro tiktok",
    "roteiro shorts", "roteiro 30", "roteiro 60", "roteiro 1 minuto",
    "hook", "legenda", "caption", "hashtag",
    # Onboarding — não precisa de modelo pesado
    "meu nicho", "meu canal", "minhas redes", "configurar perfil",
    "começar", "como funciona", "tutorial", "ajuda",
]

# ── Média — padrão quando não tem match ──────────────────────
_KEYWORDS_MEDIA = [
    "roteiro", "analise", "análise", "sugestão", "ideia", "ideias",
    "tendência", "estratégia", "melhorar", "crescer", "engajamento",
    "conteúdo", "vídeo", "post", "reels",
]


def classificar_tarefa(texto: str) -> Complexidade:
    t = texto.lower()

    # 1. Complexa tem prioridade
    for kw in _KEYWORDS_COMPLEXA:
        if kw in t:
            return Complexidade.COMPLEXA

    # 2. Simples explícito
    for kw in _KEYWORDS_SIMPLES:
        if kw in t:
            return Complexidade.SIMPLES

    # 3. Heurística por tamanho
    palavras = len(t.split())
    if palavras > 60:
        return Complexidade.COMPLEXA
    if palavras < 10:
        return Complexidade.SIMPLES

    # 4. Keywords médias
    for kw in _KEYWORDS_MEDIA:
        if kw in t:
            return Complexidade.MEDIA

    return Complexidade.MEDIA


def selecionar_modelo_texto(complexidade: Complexidade) -> dict:
    """
    SIMPLES  → llama-3.1-8b-instant  via Groq  (~2-4s)
    MEDIA    → llama-3.3-70b-versatile via Groq (~4-8s)
    COMPLEXA → gemini-1.5-pro                  (~8-15s)
    """
    configs = {
        Complexidade.SIMPLES: {
            "provedor": "groq",
            "modelo": "llama-3.1-8b-instant",
            "max_tokens": 500,
            "motivo": "Simples → Llama 3.1 8B (rápido, ~2-4s)",
        },
        Complexidade.MEDIA: {
            "provedor": "groq",
            "modelo": "llama-3.3-70b-versatile",
            "max_tokens": 900,
            "motivo": "Média → Llama 3.3 70B (~4-8s)",
        },
        Complexidade.COMPLEXA: {
            "provedor": "gemini",
            "modelo": "gemini-1.5-pro",
            "max_tokens": 1800,
            "motivo": "Complexa → Gemini 1.5 Pro (qualidade máxima)",
        },
    }
    return configs[complexidade]


def selecionar_modelo_imagem(estilo: str = "") -> dict:
    e = estilo.lower()
    if any(k in e for k in ["realista", "foto", "fotorrealista", "real"]):
        modelo = "black-forest-labs/FLUX.1-dev"
    else:
        modelo = "black-forest-labs/FLUX.1-schnell"
    return {
        "provedor": "wavespeed",
        "modelo": modelo,
        "motivo": f"Imagem '{estilo or 'padrão'}' → WaveSpeed {modelo}",
    }


def selecionar_modelo_video(tipo: str = "t2v", qualidade: str = "normal") -> dict:
    modelos = {
        ("t2v", "normal"):  {"modelo": "wavespeed-ai/wan-t2v",      "preco_seg": 0.08},
        ("t2v", "premium"): {"modelo": "wavespeed-ai/wan-t2v-480p", "preco_seg": 0.10},
        ("i2v", "normal"):  {"modelo": "wavespeed-ai/wan-i2v-720p", "preco_seg": 0.12},
        ("i2v", "premium"): {"modelo": "wavespeed-ai/wan-i2v-720p", "preco_seg": 0.12},
    }
    cfg = modelos.get((tipo, qualidade), modelos[("t2v", "normal")])
    return {
        "provedor": "wavespeed",
        **cfg,
        "motivo": f"Vídeo {tipo} {qualidade} → WaveSpeed",
    }


def log_decisao(texto: str, complexidade: Complexidade, config: dict):
    print(
        f"[ROUTER] '{texto[:50]}' → {complexidade.value.upper()} "
        f"| {config['provedor']} / {config['modelo']} "
        f"| max_tokens={config['max_tokens']}"
    )
def selecionar_modelo_imagem(complexidade): return {'provedor': 'wavespeed', 'modelo': 'flux-dev'}

def selecionar_modelo_video(complexidade): return {'provedor': 'wavespeed', 'modelo': 'wan-t2v'}
