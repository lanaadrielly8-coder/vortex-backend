"""
VORTEX — Router de modelos
Classifica a complexidade da tarefa e seleciona o melhor modelo.
"""
from enum import Enum


class Complexidade(str, Enum):
    SIMPLES   = "simples"
    MEDIA     = "media"
    COMPLEXA  = "complexa"
    CRIATIVA  = "criativa"


# Palavras-chave por complexidade
_KW_COMPLEXA = [
    "roteiro", "script", "história", "historia", "analise", "análise",
    "estratégia", "estrategia", "explica", "como funciona", "por que",
    "diferença", "diferença", "compare", "argumente", "código", "codigo",
    "programa", "desenvolve", "cria um sistema", "arquitetura",
]

_KW_CRIATIVA = [
    "viral", "hook", "tiktok", "reels", "short", "youtube", "instagram",
    "engaja", "crescimento", "algoritmo", "tendência", "tendencia",
    "copy", "headline", "título viral", "titulo viral",
]

_KW_SIMPLES = [
    "oi", "olá", "ola", "tudo bem", "bom dia", "boa tarde", "boa noite",
    "obrigado", "valeu", "ok", "certo", "entendi", "sim", "não",
]


def classificar_tarefa(texto: str) -> Complexidade:
    """Classifica a complexidade do texto para selecionar o modelo ideal."""
    t = texto.lower().strip()

    # Textos muito curtos — simples
    if len(t) < 20:
        return Complexidade.SIMPLES

    # Verificar palavras-chave
    for kw in _KW_SIMPLES:
        if t == kw or t.startswith(kw + " "):
            return Complexidade.SIMPLES

    for kw in _KW_CRIATIVA:
        if kw in t:
            return Complexidade.CRIATIVA

    for kw in _KW_COMPLEXA:
        if kw in t:
            return Complexidade.COMPLEXA

    # Texto longo — média/complexa
    if len(t) > 200:
        return Complexidade.COMPLEXA
    if len(t) > 80:
        return Complexidade.MEDIA

    return Complexidade.SIMPLES


def selecionar_modelo_texto(complexidade: Complexidade) -> dict:
    """Retorna configuração do modelo baseada na complexidade."""
    configs = {
        Complexidade.SIMPLES: {
            "provedor": "groq",
            "modelo": "llama-3.3-70b-versatile",
            "max_tokens": 800,
            "temperatura": 0.7,
        },
        Complexidade.MEDIA: {
            "provedor": "gemini",
            "modelo": "gemini-2.0-flash",
            "max_tokens": 1500,
            "temperatura": 0.8,
        },
        Complexidade.COMPLEXA: {
            "provedor": "gemini",
            "modelo": "gemini-2.0-flash",
            "max_tokens": 2500,
            "temperatura": 0.85,
        },
        Complexidade.CRIATIVA: {
            "provedor": "openrouter",
            "modelo": "deepseek/deepseek-chat-v3-0324:free",
            "max_tokens": 3000,
            "temperatura": 0.9,
        },
    }
    return configs.get(complexidade, configs[Complexidade.MEDIA])


def log_decisao(texto: str, complexidade: Complexidade, config: dict) -> None:
    """Loga a decisão de roteamento para debug."""
    texto_preview = texto[:60].replace("\n", " ")
    print(f"[ROUTER] '{texto_preview}...' → {complexidade.value} → {config['provedor']}/{config.get('modelo','?')}")