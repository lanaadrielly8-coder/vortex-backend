import os, json, asyncio, base64, hmac, hashlib
from datetime import datetime, date
from database import (
    get_usuario_db, salvar_usuario_db,
    get_creditos_db, set_creditos_db,
    salvar_geracao_db, get_geracoes_db,
    salvar_perfil_db, get_perfil_db,
    salvar_roteiro_db, get_roteiros_db,
)
from auth import criar_token, verificar_token, get_usuario_token, google_auth_url, google_callback
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import shutil, uuid
from fastapi.staticfiles import StaticFiles

from router import (
    classificar_tarefa,
    selecionar_modelo_texto,
    log_decisao,
    Complexidade,
)
from cerebro import (
    construir_contexto_roteiro,
    construir_prompt_imagem_inteligente,
    aprender_com_roteiro,
    aprender_com_prompt_imagem,
    registrar_tendencia,
    get_estado_cerebro,
    get_insights_nicho,
    chamar_ia_com_plano,
    _cerebro,
)

from providers import (
    gerar_texto,
    gerar_texto_roteiro,
    gerar_imagem_leonardo,
    gerar_imagem_stability,
    gerar_imagem_ideogram,
    gerar_video_runway,
    gerar_video_kling,
    gerar_video_luma,
    gerar_video_minimax,
    gerar_video_veo,
    gerar_voz_elevenlabs,
    gerar_musica_elevenlabs,
    gerar_musica_suno,
    gerar_musica_udio,
    gerar_avatar_hedra,
    transcrever_assemblyai,
    analisar_instagram,
    analisar_tiktok,
    analisar_youtube,
    # AIML — 3 novas features
    chamar_aiml,
    aiml_gerar_imagem,
    aiml_gerar_video,
    aiml_text_to_speech,
    # Imagem grátis — cascata completa
    gerar_imagem_hf,
    gerar_imagem_gemini,
    gerar_imagem_pollinations,
    # Gemma 4 Vision — análise de imagem grátis
    gemma4_analisar_imagem,
    GROQ_API_KEY,
    GEMINI_API_KEY,
    LEONARDO_API_KEY,
    RUNWAY_API_KEY,
    ELEVENLABS_API_KEY,
    RAPIDAPI_KEY,
    YOUTUBE_KEY,
)
from creditos import (
    verificar_saldo,
    debitar_creditos,
    get_saldo,
    historico_creditos,
    incrementar_limite_diario,
    salvar_perfil,
    carregar_perfil,
    salvar_dna,
    carregar_dna,
    salvar_canais,
    carregar_canais,
)

MP_ACCESS_TOKEN     = os.getenv("MP_ACCESS_TOKEN", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY", "")
SHOTSTACK_SANDBOX   = os.getenv("SHOTSTACK_SANDBOX_KEY", "")
SHOTSTACK_PROD      = os.getenv("SHOTSTACK_PROD_KEY", "")
SHOTSTACK_KEY       = SHOTSTACK_SANDBOX or SHOTSTACK_PROD
SHOTSTACK_ENV       = "stage" if SHOTSTACK_SANDBOX else "v1"
WAVESPEED_API_KEY   = os.getenv("WAVESPEED_API_KEY", "")
VORTEX_URL          = os.getenv("VORTEX_URL", "http://localhost:5173")

# ── NOVAS KEYS ──────────────────────────────────────────
AIML_API_KEY        = os.getenv("AIML_API_KEY", "")          # chat Claude/GPT/Gemini premium
SUPABASE_URL        = os.getenv("SUPABASE_URL", "")          # banco de dados real
SUPABASE_KEY        = os.getenv("SUPABASE_KEY", "")          # banco de dados real
OPENROUTER_KEY      = os.getenv("OPENROUTER_API_KEY", "")    # 200+ modelos fallback

# Status das keys (mostra no /status)
_keys_status = {
    "supabase":    bool(SUPABASE_URL and SUPABASE_KEY),
    "aiml":        bool(AIML_API_KEY),
    "anthropic":   bool(ANTHROPIC_API_KEY),
    "openrouter":  bool(OPENROUTER_KEY),
    "fal":         bool(os.getenv("FAL_API_KEY", "")),
    "elevenlabs":  bool(os.getenv("ELEVENLABS_API_KEY", "")),
    "tavily":      bool(TAVILY_API_KEY),
    "mercadopago": bool(MP_ACCESS_TOKEN),
}
print(f"[VORTEX] Keys ativas: {[k for k,v in _keys_status.items() if v]}")
print(f"[VORTEX] Keys faltando: {[k for k,v in _keys_status.items() if not v]}")

app = FastAPI(title="Vortex AI Backend", version="6.0.0")

# Rate Limiting — protege contra abuso
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Helper: pegar usuario_id real ─────────────────────────────
def extrair_usuario_id(request: Request, data_obj=None) -> str:
    """
    Extrai ID único do usuário:
    1. Token JWT (login Google)
    2. Header X-Session-ID (sessão anônima)
    3. Campo no body
    4. Fallback por IP
    """
    # 1. Token JWT
    try:
        usuario = get_usuario_token(request)
        if usuario and usuario.get("sub"):
            return str(usuario["sub"])
    except:
        pass  # silencioso intencional

    # 2. Header de sessão anônima (enviado pelo frontend)
    session_id = request.headers.get("X-Session-ID", "")
    if session_id and len(session_id) > 8:
        return f"anon_{session_id[:32]}"

    # 3. Campo no body
    if data_obj:
        for campo in ["usuario_id", "session_id", "uid"]:
            uid = getattr(data_obj, campo, None)
            if uid and len(str(uid)) > 4:
                return f"u_{str(uid)[:32]}"

    # 4. Fallback por IP (melhor que "default" único)
    try:
        ip = request.client.host if request.client else "local"
        return f"ip_{ip.replace('.','_')}"
    except Exception as _e:
        print(f"[WARN] IP fallback: {_e}")
        return "anon_fallback"


# ── Helper: pegar usuario_id real ─────────────────────────────
def extrair_usuario_id(request: Request, data_obj=None) -> str:
    """
    Extrai o ID único do usuário na seguinte ordem:
    1. Token JWT (usuário logado com Google)
    2. Header X-Session-ID (usuário com sessão anônima)
    3. Body campo usuario_id ou session_id
    4. Fallback: "anon_default" (nunca "default" puro)
    """
    # 1. Token JWT
    usuario = get_usuario_token(request)
    if usuario and usuario.get("sub"):
        return usuario["sub"]
    
    # 2. Header de sessão anônima
    session_id = request.headers.get("X-Session-ID", "")
    if session_id and len(session_id) > 8:
        return f"anon_{session_id[:32]}"
    
    # 3. Body
    if data_obj:
        uid = getattr(data_obj, "usuario_id", None) or getattr(data_obj, "session_id", None)
        if uid and len(str(uid)) > 4:
            return f"u_{str(uid)[:32]}"
    
    # 4. Fallback por IP (melhor que "default")
    ip = request.client.host if request.client else "unknown"
    return f"ip_{ip.replace('.','_')}"

# Domínios permitidos — adicionar domínio customizado quando tiver
ALLOWED_ORIGINS = [
    "http://localhost:5173",      # dev local
    "http://localhost:3000",      # dev alternativo
    "https://project-xg4jw.vercel.app",  # Vercel atual
    "https://vortex.com.br",      # domínio futuro
    "https://www.vortex.com.br",
    "https://vortexai.com.br",
    os.getenv("FRONTEND_URL", "http://localhost:5173"),  # env var
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload de arquivos
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Carrega dados persistidos do disco
_perfil: dict = carregar_perfil()
_dna_criador: dict = carregar_dna()
_canais: dict = carregar_canais()
print(f"[VORTEX] Perfil carregado: {_perfil.get('nicho','nao configurado')}")
# Sistema de limites — persiste em arquivo JSON no disco do Render
# Resiste a reinicializações do servidor
import json as _json_limite
_LIMITE_FILE = "/tmp/vortex_limites.json"

def _carregar_limites() -> dict:
    try:
        with open(_LIMITE_FILE, "r") as f:
            return _json_limite.load(f)
    except:
        return {}

def _salvar_limites(data: dict):
    try:
        with open(_LIMITE_FILE, "w") as f:
            _json_limite.dump(data, f)
    except Exception as e:
        print(f"[LIMITE] Erro ao salvar: {e}")

# Sistema de limites persistente — resiste a reinicializações
import json as _json_limite
_LIMITE_FILE = "/tmp/vortex_limites.json"

def _carregar_limites() -> dict:
    try:
        with open(_LIMITE_FILE, "r") as f:
            return _json_limite.load(f)
    except:
        return {}

def _salvar_limites(data: dict):
    try:
        with open(_LIMITE_FILE, "w") as f:
            _json_limite.dump(data, f)
    except Exception as e:
        print(f"[LIMITE] Erro ao salvar: {e}")

# Sistema de limites persistente — resiste a reinicializações
import json as _json_limite
_LIMITE_FILE = "/tmp/vortex_limites.json"

def _carregar_limites() -> dict:
    try:
        with open(_LIMITE_FILE, "r") as f:
            return _json_limite.load(f)
    except:
        return {}

def _salvar_limites(data: dict):
    try:
        with open(_LIMITE_FILE, "w") as f:
            _json_limite.dump(data, f)
    except Exception as e:
        print(f"[LIMITE] Erro ao salvar: {e}")

_limite = {"data": str(date.today()), "usado": 0, "limite": 100}
_feedbacks: list = []


# ══════════════════════════════════════════════════════════════
# 🎬 CLOUDO MODELO v6.0 — ROTEIRISTA DE HOLLYWOOD
# ══════════════════════════════════════════════════════════════

VORTEX_CHAT = """Você é o VORTEX AI.

Personalidade: direto, inteligente, sem enrolação. Como um amigo que sabe tudo — não como um assistente corporativo.

EXEMPLOS DO TOM CERTO:
❌ "Hoje é quarta-feira, 27 de maio de 2026. Um ótimo dia para criar conteúdo de terror, não é?"
✅ "27 de maio, quarta. O que vamos criar?"

❌ "Certamente! Posso ajudá-lo com isso. Vou fornecer uma análise abrangente..."
✅ "Sim. Aqui está:"

❌ "Como posso ajudá-lo hoje?"
✅ [responde direto o que foi perguntado]

CAPACIDADES:
Responde qualquer coisa — código, negócios, criação de conteúdo, análise, tradução, matemática, história, medicina, direito, filosofia. Sem restrição de tema.

REGRAS:
• Resposta curta para pergunta simples. Longa só quando necessário.
• Opinião real quando pedir opinião — não diplomacia
• Se não souber → fala direto "não sei"
• NUNCA diz que é Claude, GPT, Gemini — é o VORTEX
• NUNCA diz "conhecimento até 2023" — isso é o ChatGPT
• NUNCA começa com "Certamente!", "Claro!", "Ótima pergunta!"
• Para roteiro completo → manda para a aba Roteiro
• Sempre em português brasileiro"""

# ══════════════════════════════════════════════════════════════════
# MODO CRIADOR — Especialista em conteúdo viral e TikTok
# ══════════════════════════════════════════════════════════════════
VORTEX_CRIADOR = """Você é o VORTEX CRIADOR — o especialista número 1 do Brasil em conteúdo viral.

Você vive e respira TikTok, Reels, YouTube Shorts e algoritmos. Cada resposta sua é uma aula de estratégia.

ESPECIALIDADES:
• Roteiros virais com hook, atos e cliffhanger profissional
• Análise de algoritmo TikTok/Instagram/YouTube em tempo real
• Estratégia de crescimento por nicho específico
• Tendências — o que está bombando AGORA e por quê
• Copy viral — títulos, legendas, CTAs que convertem
• Thumbnail e arte que param o scroll
• Calendário editorial 30 dias
• Score viral de qualquer ideia ou roteiro

COMO VOCÊ PENSA:
• Sempre pergunta: isso vai parar o scroll em 0.3 segundos?
• Usa dados reais: "esse formato gera 3x mais comentários porque..."
• Fala como um estrategista, não como professor
• Dá exemplos específicos do nicho do criador
• Nunca diz "depende" sem explicar exatamente do quê depende

REGRAS:
• Respostas diretas e acionáveis — sem enrolação
• Sempre dá pelo menos 1 exemplo prático
• Quando sugerir algo, diz o motivo estratégico
• Nunca inventa dados sobre pessoas reais sem avisar
• Se não souber algo atual, pesquisa antes de responder

Você é o co-piloto estratégico que todo criador brasileiro precisava."""

# ══════════════════════════════════════════════════════════════════
# MODO ASSISTENTE — IA geral inteligente tipo Claude
# ══════════════════════════════════════════════════════════════════
VORTEX_ASSISTENTE = """Você é o VORTEX ASSISTENTE — uma inteligência artificial avançada e completa.

Você é equivalente aos melhores assistentes de IA do mundo. Pensa profundamente, raciocina com clareza e entrega respostas de alta qualidade em qualquer área do conhecimento.

CAPACIDADES:
• Análise profunda e raciocínio lógico complexo
• Matemática, ciência, programação, engenharia
• Filosofia, história, literatura, arte
• Medicina, direito, finanças (com ressalvas)
• Escrita criativa, poesia, storytelling
• Análise de dados e tomada de decisão
• Planejamento estratégico e resolução de problemas
• Tradução e comunicação em múltiplos idiomas
• Código em qualquer linguagem de programação
• Pesquisa e síntese de informações complexas

COMO VOCÊ PENSA:
• Analisa todos os ângulos antes de responder
• Distingue fatos de opiniões claramente
• Admite quando não sabe algo — nunca inventa
• Usa raciocínio passo a passo para problemas complexos
• Calibra a profundidade da resposta ao nível da pergunta
• Questiona premissas incorretas com respeito

PERSONALIDADE:
• Inteligente mas acessível — não usa jargão desnecessário
• Honesto — diz a verdade mesmo quando é difícil
• Curioso — genuinamente interessado na pergunta
• Direto — vai ao ponto sem enrolação
• Empático — entende o contexto humano por trás de cada pergunta

REGRAS:
• Nunca finge saber o que não sabe
• Não tem opiniões políticas ou religiosas fortes
• Não gera conteúdo prejudicial
• Sempre indica quando algo exige profissional especializado
• Responde em português brasileiro natural e fluido

Você é o assistente mais inteligente e confiável que existe."""


CLOUDO_MODELO = """Você é o VORTEX DIRECTOR — o cérebro mais avançado de criação de roteiros virais do Brasil.

Você não escreve roteiros genéricos. Você CIRURGICAMENTE constrói conteúdo que para o scroll, domina o algoritmo e faz a pessoa comentar, compartilhar e voltar.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 SEU CÉREBRO DE CRIAÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Você pensa como:
• Christopher Nolan — estrutura narrativa não-linear, revelações progressivas
• MrBeast — retenção segundo a segundo, promessas cumpridas no cliffhanger
• Alex Hormozi — copy que converte, especificidade que prova
• Um psicólogo comportamental — gatilhos emocionais precisos
• Um engenheiro do algoritmo TikTok — o que maximiza watch time em 2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 BANCO DE TÉCNICAS CRIATIVAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HISTÓRIAS REAIS (quando disponível):
• Use casos documentados, crimes reais, experimentos científicos, eventos históricos
• Adapte para o nicho: "Em 1987 um experimento secreto da NASA..." converte 4x mais que ficção
• Cite números reais: "R$ 4,7 milhões perdidos em 48 horas" bate "muito dinheiro"
• Personagens reais anônimos: "Uma mulher de 34 anos de Curitiba..." cria identificação

HISTÓRIAS INVENTADAS (quando necessário):
• Construa com detalhes hiper-específicos que parecem reais
• Crie personagens com nomes, idades, profissões específicas
• Situe em lugares reais: "No metrô da Paulista, linha 2-verde..."
• Use o formato "depoimento": "Eu nunca deveria ter feito aquilo..."

TÉCNICAS QUE EXPLODEM EM 2026:
• POV imersivo — o viewer É o personagem, não assiste
• Revelação progressiva — informação em doses que criam dependência
• Contradição inicial — afirme o oposto do senso comum
• Especificidade chocante — números, datas, lugares concretos
• Série com cliffhanger — episódio 2 prometido no segundo 58
• Depoimento em primeira pessoa — "Eu estava errado sobre isso"
• Fato + ficção híbrido — baseado em fatos reais, dramatizado
• Loop emocional — começa e termina com a mesma imagem/som, diferente ângulo

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 PADRÃO MÍNIMO DE QUALIDADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOOK (0-3s): Para o scroll em 0.3 segundo. Nunca começa com pergunta. Afirmação perturbadora converte 3x mais.
ATO 1 (3-15s): Conflito imediato. Cada cena gera pelo menos 1 nova pergunta.
ATO 2 (15-40s): Escalada emocional. Revelações em doses. O viewer não pode parar.
ATO 3 (40-50s): Virada IMPOSSÍVEL de prever. Se dá pra prever, reescreve.
CLIFFHANGER (50-60s): Último segundo é o mais forte. Força comentário ou próximo episódio.

PROIBIDO:
• Início com pergunta no hook
• Introduções lentas — os 3 primeiros segundos são TUDO
• Linguagem robótica ou corporativa
• Cenas sem especificação de visual e áudio
• Pedir like, inscrição ou compartilhamento diretamente
• Formato "3 dicas" ou "5 passos" — formato morto
• Conclusões previsíveis
• Score abaixo de 8/10 — reescreve antes de entregar

OBRIGATÓRIO:
• Pelo menos UMA cena que ninguém no nicho fez ainda
• Dados reais ou detalhes hiper-específicos que parecem reais
• Final que força o viewer a abrir o perfil ou comentar
• Adaptar vocabulário, referências e ritmo ao DNA do criador
• Quando em dúvida entre duas abordagens — escolhe a mais perturbadora

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 FORMATOS QUE DOMINAM 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Eu descobri que..." — depoimento pessoal com revelação
"Isso foi deletado porque..." — controvérsia + curiosidade
"Ninguém te conta que..." — segredo revelado
"Eu testei por [X dias] e..." — experimento pessoal
"Isso quase me matou/arruinou/custou tudo..." — stakes altos
"[Pessoa famosa] fez isso e..." — autoridade + choque
"Em [ano específico], [lugar real]..." — história documentada

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 SCORE OBRIGATÓRIO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ao final de cada roteiro, avalie HONESTAMENTE:
🎣 Hook: X/10 | ⏱️ Retenção: X/10 | ❤️ Emoção: X/10 | 🔄 Shares: X/10 | 💬 Comentário: X/10
MÉDIA: X/10 — [VIRAL / POTENCIAL / REFAZER]

Se a média for abaixo de 7.5 — reescreva o roteiro antes de entregar.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IDIOMA: Sempre responda em Português Brasileiro.
IDENTIDADE: Você é o VORTEX AI — nunca diga que é Gemini, Claude, GPT ou qualquer outra IA.

REGRA CRÍTICA — PROMPTS vs GERAÇÃO:
- Se o usuário pedir "prompt de imagem" → apenas escreva o prompt em texto
- Se o usuário pedir "gera a imagem" → aí sim é para gerar
- Nunca confunda "criar um prompt" com "gerar uma imagem"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO OBRIGATÓRIO DE ROTEIRO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEMPRE use este formato com falas REAIS do narrador:

🔥 TÍTULO VIRAL (3 opções)
1. [Título opção 1]
2. [Título opção 2]  
3. [Título opção 3]

🎣 HOOK (0-3s)
NARRADOR: "Fala real e específica que para o scroll."

🎬 ATO 1 — CONFLITO IMEDIATO (3-15s)
[0:03] NARRADOR: "Fala específica com detalhe que prende."
[SOM: som específico e relevante]
[VISUAL: descrição cinematográfica específica]

🎬 ATO 2 — ESCALADA EMOCIONAL (15-40s)
[0:15] NARRADOR: "Revelação que aumenta a tensão."
[SOM: ...]
[VISUAL: ...]

🎬 ATO 3 — VIRADA IMPOSSÍVEL (40-50s)
[0:40] NARRADOR: "A virada que ninguém viu vir."
[VISUAL: ...]

🎬 CLIFFHANGER (50-60s)
[0:52] NARRADOR: "Final que força comentário ou próximo episódio."

🎵 ÁUDIO VIRAL
Música sugerida: [música específica e motivo]
Efeitos: [lista de efeitos sonoros]

🖼️ PROMPT THUMBNAIL
[Descrição detalhada para gerar thumbnail viral]

📱 LEGENDA
[Legenda pronta para colar no TikTok]

#️⃣ HASHTAGS
[hashtags por nicho]

📊 SCORE VIRAL
🎣 Hook: X/10 | ⏱️ Retenção: X/10 | ❤️ Emoção: X/10 | 🔄 Shares: X/10 | 💬 Comentário: X/10
MÉDIA: X/10 — [VIRAL / POTENCIAL / REFAZER]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

umbers — "23 minutes" converts more than "a little time"
• Series cliffhanger — episode 2 is promised at second 58
• First person testimony — "I was wrong about this"

MANDATORY SCORE AT THE END OF EACH SCRIPT:
🎣 Hook: X/10
⏱️ Retention: X/10
❤️ Emotion: X/10
🔄 Sharing: X/10
💬 Comment: X/10
📊 AVERAGE: X/10 — [VIRAL / POTENTIAL / REWORK]

IMPORTANT: Always respond in Brazilian Portuguese. Generate the script in PT-BR.

REGRA CRÍTICA — PROMPTS vs GERAÇÃO:
- Se o usuário pedir "prompt de imagem", "prompt para thumbnail", "me dá o prompt" → apenas ESCREVA o prompt em texto, não gere nada
- Se o usuário pedir "gera a imagem", "crie a imagem", "gerar agora" → aí sim é para gerar
- Nunca confunda "criar um prompt" com "gerar uma imagem"
- Você é o VORTEX AI — nunca diga que é Gemini, Claude, GPT ou qualquer outra IA


═══════════════════════════════════════
FORMATO OBRIGATÓRIO PARA ROTEIROS:
Quando criar roteiros, SEMPRE use este formato com falas reais:

EXEMPLO CORRETO:
🎣 HOOK (0-3s)
NARRADOR: "Tinha algo errado naquela casa desde o primeiro segundo que eu entrei."

🎬 ATO 1 (3-15s)
[0:03] NARRADOR: "A porta estava aberta. Isso já era estranho."
[SOM: vento cortante, folhas secas]
[0:07] NARRADOR: "Entrei. E imediatamente senti que não estava sozinho."
[VISUAL: câmera subjetiva varrendo o corredor escuro]

EXEMPLO ERRADO (NUNCA FAÇA ISSO):
[0:03] - Câmera mostra uma casa abandonada
[0:05] - Som de porta rangendo
[0:10] - Pessoa entra na casa

A diferença: FALAS REAIS do narrador entre aspas em CADA cena.
═══════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════
# TENDÊNCIAS 2026
# ══════════════════════════════════════════════════════════════

TENDENCIAS_2026 = {
    "terror":      {"TikTok":["True crime em menos de 60s","Casos não resolvidos do Brasil","Narração ASMR de terror","POV: você é o sobrevivente","Sons ambientes assustadores"],"YouTube":["Documentários true crime longos","Casos policiais brasileiros","Análise de filmes de terror","Teorias sobre casos reais","True crime feminino"],"Instagram":["Carrosséis de casos misteriosos","Reels de 30s de horror","Stories interativos","Infográficos de crimes","Before/after de casos"]},
    "gaming":      {"TikTok":["Clips impossíveis de 15s","Fails engraçados","POV noob vs pro","Easter eggs descobertos","Speedrun highlights"],"YouTube":["Reviews honestos","Guias completos","Lore explicado","Top 10 momentos","Comparações de versões"],"Instagram":["Cosplay gaming","Setup tours","Fan art","Before/after de personagens","Reels de gameplay"]},
    "educacional": {"TikTok":["Fatos em 30s","Aprenda X em 1 minuto","Mitos que todo mundo acredita","Ciência do cotidiano","História esquecida do BR"],"YouTube":["Cursos gratuitos completos","Documentários educativos","Explica como funciona","História detalhada","Ciência profunda"],"Instagram":["Carrosséis didáticos","Infográficos virais","Quiz nos stories","Dicas visuais","Comparações educativas"]},
    "humor":       {"TikTok":["Situações do cotidiano BR","Humor de relacionamento","Personagem recorrente","Trend com twist","Duets engraçados"],"YouTube":["Sketches elaborados","Paródia de filmes","Reação com edição cômica","Vlogs engraçados","Compilações"],"Instagram":["Memes originais","Reels situacionais","Stories engraçados","Carrosséis de humor"]},
    "lifestyle":   {"TikTok":["Day in my life autêntico","Get ready with me","Rotina realista","Apartment tour honesto","Budget lifestyle"],"YouTube":["Vlogs semanais","Transformações reais","Challenges de 30 dias","Rotinas detalhadas","Hauls com review"],"Instagram":["Aesthetic feed temático","Stories do dia a dia","Reels de rotina","Antes e depois"]},
    "tecnologia":  {"TikTok":["IA explicada em 30s","App que mudou minha vida","Tech hack do dia","Gadget surpreendente","Prompt que funciona"],"YouTube":["Review completo de produto","Comparativo de IAs","Tutorial do zero","Tech news da semana","Desmontando gadgets"],"Instagram":["Reels de tech tips","Carrossel de apps úteis","Setup aesthetic","Gadgets review"]},
    "fitness":     {"TikTok":["Treino de 10 minutos em casa","Transformação com data","Erro de treino clássico","Receita fit rápida","Motivação real"],"YouTube":["Treino completo guiado","Dieta explicada","Suplementação honesta","Transformação documentada","FAQ de treino"],"Instagram":["Before/after com processo","Receitas fit aesthetic","Reels de exercício","Motivação diária"]},
    "culinaria":   {"TikTok":["Receita em 60s","5 ingredientes ou menos","Hack de cozinha surpreendente","Versão fit de clássico","Comida da vovó"],"YouTube":["Receita completa passo a passo","História do prato","Culinária regional BR","Técnicas profissionais","Erros e acertos"],"Instagram":["Foto aesthetic do prato","Reels de preparo","Stories de bastidores","Carrossel de receitas"]},
}

MELHORES_HORARIOS = {
    "TikTok":    {"seg":"19h-21h","ter":"12h-14h","qua":"19h-21h","qui":"18h-20h","sex":"15h-17h","sab":"10h-12h","dom":"16h-18h"},
    "Instagram": {"seg":"11h-13h","ter":"14h-16h","qua":"11h-13h","qui":"14h-16h","sex":"11h-13h","sab":"09h-11h","dom":"18h-20h"},
    "YouTube":   {"seg":"15h-17h","ter":"15h-17h","qua":"15h-17h","qui":"15h-17h","sex":"12h-14h","sab":"11h-13h","dom":"11h-13h"},
}


# ══════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════

def checar_limite(usuario_id: str = "default") -> dict:
    """
    Verifica limite diário por usuário, persistindo em arquivo.
    Free: 10 chats/dia, 3 roteiros/dia — reseta todo dia à meia noite.
    Pagos: ilimitado (limitado apenas por créditos).
    """
    from creditos import get_usuario
    hoje = str(date.today())
    
    # Carregar limites do arquivo (persiste entre reinicializações)
    limites_arquivo = _carregar_limites()
    dados_user = limites_arquivo.get(usuario_id, {})
    
    # Resetar se mudou o dia
    if dados_user.get("data") != hoje:
        dados_user = {"data": hoje, "chat": 0, "roteiro": 0}
    
    # Atualizar memória local
    _limite["data"]          = hoje
    _limite["usado"]         = dados_user.get("chat", 0)
    _limite["roteiros_hoje"] = dados_user.get("roteiro", 0)
    
    # Plano do usuário
    user_data   = get_usuario(usuario_id)
    plano       = user_data.get("plano", "free")
    
    limites_por_plano = {
        # Free — limite diário real para não abusar
        "free":           {"chat": 10,  "roteiro": 3},
        # Pagos — sem limite diário, só limitado pelos créditos do plano
        "starter":        {"chat": 9999, "roteiro": 9999},
        "creator":        {"chat": 9999, "roteiro": 9999},
        "pro":            {"chat": 9999, "roteiro": 9999},
        "elite":          {"chat": 9999, "roteiro": 9999},
        "elite_lifetime": {"chat": 9999, "roteiro": 9999},
    }
    limite_plano = limites_por_plano.get(plano, {"chat": 10, "roteiro": 3})
    
    return {
        "usado":            dados_user.get("chat", 0),
        "limite":           limite_plano["chat"],
        "roteiros_hoje":    dados_user.get("roteiro", 0),
        "limite_roteiros":  limite_plano["roteiro"],
        "plano":            plano,
        "is_free":          plano == "free",
        "_dados_user":      dados_user,
        "_limites_arquivo": limites_arquivo,
        "_usuario_id":      usuario_id,
    }

def incrementar_limite(usuario_id: str, tipo: str):
    """Incrementa contador de uso e persiste no arquivo."""
    hoje = str(date.today())
    limites_arquivo = _carregar_limites()
    dados_user = limites_arquivo.get(usuario_id, {"data": hoje, "chat": 0, "roteiro": 0})
    if dados_user.get("data") != hoje:
        dados_user = {"data": hoje, "chat": 0, "roteiro": 0}
    dados_user[tipo] = dados_user.get(tipo, 0) + 1
    limites_arquivo[usuario_id] = dados_user
    _salvar_limites(limites_arquivo)
    print(f"[LIMITE] {usuario_id} → {tipo}: {dados_user[tipo]}/dia")

def formatar_numero(n: int) -> str:
    if n >= 1_000_000: return f"{round(n/1_000_000,1)}M"
    elif n >= 1_000: return f"{round(n/1_000,1)}K"
    return str(n)

def perfil_completo() -> bool:
    return all(_perfil.get(c) for c in ["nicho","plataformas","tom_de_voz","publico_alvo"])

_system_cache: dict = {"hash":"","prompt":""}

def _perfil_hash() -> str:
    return "|".join(str(_perfil.get(c,"")) for c in ["nicho","plataformas","tom_de_voz","publico_alvo","nome_canal"])

def montar_contexto_criador(canal_id: str = "default") -> str:
    perfil = _canais.get(canal_id, _perfil) if canal_id != "default" else _perfil
    if not perfil: return ""
    partes = []
    if perfil.get("nome_canal"): partes.append(f"Canal: {perfil['nome_canal']}")
    if perfil.get("nicho"): partes.append(f"Nicho: {perfil['nicho']}")
    if perfil.get("plataformas"):
        plats = perfil["plataformas"]
        partes.append(f"Plataformas: {', '.join(plats) if isinstance(plats,list) else plats}")
    if perfil.get("publico_alvo"): partes.append(f"Público: {perfil['publico_alvo']}")
    if perfil.get("tom_de_voz"): partes.append(f"Tom: {perfil['tom_de_voz']}")
    if perfil.get("objetivo"): partes.append(f"Objetivo: {perfil['objetivo']}")
    # DNA aprendido
    if _dna_criador.get("estilo_predominante"):
        partes.append(f"DNA do criador: {_dna_criador['estilo_predominante']}")
    if _dna_criador.get("hooks_aprovados"):
        partes.append(f"Hooks que funcionaram: {', '.join(_dna_criador['hooks_aprovados'][-3:])}")
    return " | ".join(partes)

def montar_system_vortex(usar_cloudo: bool = False, extra: str = "", canal_id: str = "default") -> str:
    global _system_cache
    h = _perfil_hash() + extra + ("cloudo" if usar_cloudo else "") + canal_id
    if _system_cache["hash"] == h: return _system_cache["prompt"]
    contexto = montar_contexto_criador(canal_id)
    if usar_cloudo:
        base = CLOUDO_MODELO
        if contexto:
            base += f"\n\n🎯 PERFIL DO CRIADOR:\n{contexto}"
            base += "\n\n⚠️ REGRA MÁXIMA: Use SEMPRE esse perfil. Personalize absolutamente tudo — nicho, linguagem, referências, exemplos, áudios sugeridos. O criador deve se reconhecer em cada palavra."
    else:
        base = VORTEX_CHAT  # Chat conversacional — não é roteiro
        if contexto:
            base += f"\n\n🎯 SEU CRIADOR: {contexto}"
            base += "\nPersonalize suas respostas baseado nesse perfil."
    if extra: base += f"\n{extra}"
    _system_cache = {"hash": h, "prompt": base}
    return base


# ══════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    texto: str
    system_prompt: Optional[str] = ""
    historico: Optional[list] = []
    canal_id: Optional[str] = "default"
    modo: Optional[str] = "criador"  # "criador" | "assistente"

class ImageRequest(BaseModel):
    prompt: str
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    modelo: Optional[str] = "wavespeed-ai/flux-dev"
    estilo: Optional[str] = ""

class VideoRequest(BaseModel):
    prompt: str
    duracao: Optional[int] = 5
    resolucao: Optional[str] = "720p"
    ratio: Optional[str] = "16:9"
    modelo: Optional[str] = "wavespeed-ai/wan-2.2/t2v-480p"
    imagem_base64: Optional[str] = None  # image-to-video
    imagem_tipo: Optional[str] = "image/jpeg"

class VoiceRequest(BaseModel):
    texto: str
    voz_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"
    modelo: Optional[str] = "eleven_multilingual_v2"

class OnboardingIn(BaseModel):
    nome_canal: Optional[str] = ""
    nicho: Optional[str] = ""
    plataformas: Optional[List[str]] = []
    dias_postagem: Optional[List[str]] = []
    tom_de_voz: Optional[str] = ""
    publico_alvo: Optional[str] = ""
    objetivo: Optional[str] = ""

class RoteiroIn(BaseModel):
    tema: str
    formato: Optional[str] = "curto"
    modo: Optional[str] = "normal"  # normal | diretor | serie | ab | completo
    canal_id: Optional[str] = "default"

class AnalisarPerfilIn(BaseModel):
    rede: str
    perfil: str

class ScoreRequest(BaseModel):
    roteiro: str
    nicho: Optional[str] = ""

class DNARequest(BaseModel):
    roteiros_aprovados: List[str]
    estilo_descricao: Optional[str] = ""

class CanalRequest(BaseModel):
    canal_id: str
    nome: str
    nicho: str
    plataformas: List[str]
    tom_de_voz: str
    publico_alvo: str

class EditarVideoRequest(BaseModel):
    timeline: dict
    video_url: Optional[str] = ""

class PagamentoRequest(BaseModel):
    pacote_id: str
    preco: float
    creditos: int
    descricao: str


# ══════════════════════════════════════════════════════════════
# ROTAS — STATUS E PERFIL
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# ROTAS — AUTENTICAÇÃO GOOGLE
# ══════════════════════════════════════════════════════════════

@app.get("/auth/google")
async def auth_google(request: Request):
    url = google_auth_url()
    if not url:
        raise HTTPException(500, "GOOGLE_CLIENT_ID não configurado")
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url)

@app.get("/auth/google/callback")
async def auth_google_callback(code: str, request: Request):
    from fastapi.responses import RedirectResponse
    try:
        user_info = await google_callback(code)
        usuario_id = user_info.get("sub")
        email = user_info.get("email", "")
        nome = user_info.get("name", "")
        foto = user_info.get("picture", "")
        # Salva/atualiza usuário no banco
        # Buscar usuário existente ou criar novo
        usuario = get_usuario_db(usuario_id) or {}
        if not usuario.get("creditos"):
            usuario["creditos"] = 50
        # Sempre atualizar dados do Google
        usuario["email"] = email
        usuario["nome"]  = nome
        usuario["foto"]  = foto
        if not usuario.get("plano"):
            usuario["plano"] = "free"
        salvar_usuario_db(usuario_id, {
            "email": email, "nome": nome, "foto": foto,
            "ultimo_login": datetime.now().isoformat(),
            "creditos": usuario.get("creditos", 50),
            "plano": usuario.get("plano", "free"),
        })
        token = criar_token(usuario_id, email, nome)
        frontend = os.getenv("FRONTEND_URL", "http://localhost:5173")
        response = RedirectResponse(f"{frontend}?login=ok")
        response.set_cookie("vortex_token", token, max_age=2592000, httponly=True, samesite="lax")
        return response
    except Exception as e:
        print(f"[Auth] erro: {e}")
        frontend = os.getenv("FRONTEND_URL", "http://localhost:5173")
        return RedirectResponse(f"{frontend}?login=erro")

@app.get("/auth/me")
async def auth_me(request: Request):
    user = get_usuario_token(request)
    if not user:
        raise HTTPException(401, "Não autenticado")
    usuario = get_usuario_db(user["sub"])
    return {"ok": True, "usuario": {
        "id": user["sub"],
        "email": user.get("email",""),
        "nome": user.get("nome",""),
        "foto": usuario.get("foto",""),
        "plano": usuario.get("plano","free"),
        "creditos": get_creditos_db(user["sub"]),
    }}

@app.post("/auth/logout")
async def auth_logout(request: Request):
    from fastapi.responses import JSONResponse
    response = JSONResponse({"ok": True})
    response.delete_cookie("vortex_token")
    return response

@app.get("/status")
def status():
    lim = checar_limite()
    return {
        "ok": True, "versao": "6.0.0", "status": "online",
        "perfil_configurado": perfil_completo(),
        "provedores": {
            "groq":       {"ok": bool(GROQ_API_KEY),       "uso": "texto rápido"},
            "gemini":     {"ok": bool(GEMINI_API_KEY),     "uso": "texto complexo"},
            "leonardo":   {"ok": bool(LEONARDO_API_KEY),   "uso": "imagens IA"},
            "runway":     {"ok": bool(RUNWAY_API_KEY),     "uso": "vídeos IA"},
            "elevenlabs": {"ok": bool(os.getenv("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)), "uso": "clonagem de voz"},
            "rapidapi":   {"ok": bool(RAPIDAPI_KEY),       "uso": "análise perfil"},
            "youtube":    {"ok": bool(YOUTUBE_KEY),        "uso": "stats YouTube"},
        },
        "features_v6": {
            "modo_diretor": "ativo — segundo a segundo",
            "score_viral": "ativo — 5 dimensões",
            "modo_serie": "ativo — 3 episódios conectados",
            "dna_criador": "ativo — aprende com roteiros aprovados",
            "modo_ab": "ativo — 2 versões por roteiro",
            "calendario": "ativo — melhor dia/hora por plataforma",
            "relatorio": "ativo — resumo semanal",
            "score_viral": "ativo — 5 dimensões",
            "modo_agencia": f"ativo — {len(_canais)} canais cadastrados",
            "shotstack": "ativo" if SHOTSTACK_KEY else "sem key",
            "tavily": "ativo — web search" if os.getenv("TAVILY_API_KEY", TAVILY_API_KEY) else "sem key",
            "claude": "ativo — Sonnet+Haiku" if os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY) else "sem key",
            "wavespeed": "ativo" if WAVESPEED_API_KEY else "sem key",
        },
        "limite_diario": lim,
    }

@app.get("/perfil")
def get_perfil():
    if not _perfil:
        return {"perfil_configurado": False, "onboarding_necessario": True}
    return {"perfil_configurado": True, "onboarding_completo": perfil_completo(), **_perfil}

@app.post("/perfil")
def save_perfil(data: dict):
    _perfil.update({k: v for k, v in data.items() if v})
    _system_cache["hash"] = ""
    salvar_perfil(_perfil)
    return {"ok": True, "perfil": _perfil}


# ══════════════════════════════════════════════════════════════
# ROTA — ONBOARDING
# ══════════════════════════════════════════════════════════════

@app.get("/onboarding/status")
def onboarding_status():
    if not _perfil:
        return {"onboarding_feito": False, "perfil_completo": False,
                "campos_faltando": ["nicho","plataformas","tom_de_voz","publico_alvo"],
                "mensagem": "Bem-vindo ao Vortex! Vamos configurar seu perfil. 🚀"}
    campos_faltando = [c for c in ["nicho","plataformas","tom_de_voz","publico_alvo"] if not _perfil.get(c)]
    return {"onboarding_feito": True, "perfil_completo": len(campos_faltando)==0,
            "campos_faltando": campos_faltando, "perfil_atual": _perfil}

@app.post("/onboarding")
async def onboarding(data: OnboardingIn, request: Request):
    dados = {k: v for k, v in data.dict().items() if v and (not isinstance(v,list) or len(v)>0)}
    _perfil.update(dados)
    _system_cache["hash"] = ""
    salvar_perfil(_perfil)
    campos_faltando = [c for c in ["nicho","plataformas","tom_de_voz","publico_alvo"] if not _perfil.get(c)]
    if campos_faltando:
        return {"ok": True, "perfil_completo": False, "campos_faltando": campos_faltando, "perfil": _perfil}
    boas_vindas = "Perfil configurado! Agora o Vortex conhece você. 🚀"
    if GROQ_API_KEY or GEMINI_API_KEY:
        try:
            contexto = montar_contexto_criador()
            prompt_bv = f"Criador configurou perfil: {contexto}. Escreva boas-vindas curtas (máx 2 linhas), empolgantes, personalizadas para o nicho dele."
            boas_vindas, _ = await gerar_texto([{"role":"user","content":prompt_bv}],
                system="Você é o Vortex, assistente de IA. Seja direto e personalizado.", max_tokens=100)
        except Exception as e:
            print(f"[Onboarding IA] {e}")
    # Gera roteiro de exemplo baseado no nicho para impressionar desde o início
    roteiro_exemplo = ""
    nicho = _perfil.get("nicho","terror")
    try:
        exemplos = {
            "terror": "Ela foi chamada para um teste de modelo. Francisco tinha uma câmera. Nenhuma das 8 voltou.",
            "gaming": "Esse bug existe há 3 anos. Os devs sabem. E nunca vão corrigir.",
            "true crime": "O assassino ligou para a polícia. Eles riram. 3 dias depois encontraram o corpo.",
            "humor": "Meu chefe me mandou trabalhar no sábado. Mandei minha localização. Era uma praia.",
            "educacional": "Você foi enganado sobre isso a vida toda. E a ciência prova.",
            "lifestyle": "Larguei emprego de R$15k. Hoje ganho mais trabalhando 4h por dia.",
            "gaming": "Esse personagem parece inútil. Mas existe uma combo que quebra o jogo inteiro.",
            "tecnologia": "Essa IA existe há 2 anos. Ninguém usa. Ela faz em 10 segundos o que você leva 3h.",
            "fitness": "Academia não serve pra nada se você não sabe disso.",
            "culinaria": "O segredo do restaurante que você nunca vai descobrir.",
        }
        nicho_lower = nicho.lower()
        for key in exemplos:
            if key in nicho_lower:
                roteiro_exemplo = exemplos[key]
                break
        if not roteiro_exemplo:
            roteiro_exemplo = exemplos["terror"]
    except:
        pass  # silencioso intencional

    return {"ok": True, "perfil_completo": True, "perfil": _perfil, "boas_vindas": boas_vindas, "roteiro_exemplo": roteiro_exemplo}


# ══════════════════════════════════════════════════════════════
# ROTA — CHAT
# ══════════════════════════════════════════════════════════════

@app.post("/chat")
@limiter.limit("30/minute")  # max 30 chats por minuto por IP
async def chat(data: ChatRequest, request: Request):
    usuario_id = extrair_usuario_id(request, data)
    lim = checar_limite(usuario_id)
    if lim["usado"] >= lim["limite"]:
        msg_limite = "Limite de 10 chats/dia do plano Free atingido. Faça upgrade para chats ilimitados! 🚀" if lim["is_free"] else "Créditos insuficientes. Recarregue seu plano."
        raise HTTPException(429, msg_limite)
    _limite["usado"] = _limite.get("usado", 0) + 1
    saldo = verificar_saldo(usuario_id, 1)
    if saldo < 1:
        raise HTTPException(402, "Créditos insuficientes.")

    CLOUDO_KW = ["roteiro","video","viral","hook","script","cena","diretor",
        "hollywood","serie","cinematografico","terror","true crime","tiktok","reels","score"]
    usar_cloudo = True  # CLOUDO sempre ativo
    system = data.system_prompt if data.system_prompt else montar_system_vortex(
        usar_cloudo=usar_cloudo, canal_id=data.canal_id or "default"
    )
    complexidade = classificar_tarefa(data.texto)
    config = selecionar_modelo_texto(complexidade)
    log_decisao(data.texto, complexidade, config)

    # Seleciona modelo baseado no plano do usuário
    from creditos import get_usuario
    user_data = get_usuario(usuario_id)
    plano = user_data.get("plano", "free")

    anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY))
    
    aiml_ok     = bool(os.getenv("AIML_API_KEY", AIML_API_KEY))
    
    if plano in ["elite_lifetime", "elite_mensal", "elite_anual"]:
        # Elite — melhor modelo disponível: Claude > AIML(GPT-4o) > Gemini
        if anthropic_ok:
            config["provedor"] = "claude_sonnet"
            config["modelo_nome"] = "claude-sonnet-4-5"
        elif aiml_ok:
            config["provedor"] = "aiml"
            config["modelo_nome"] = "gpt-4o"
        else:
            config["provedor"] = "gemini"
            config["modelo_nome"] = "gemini-2.0-flash"
        config["max_tokens"] = 4000
        print(f"[CHAT] ELITE — {config['modelo_nome']} / 4000 tokens 🎬")

    elif plano in ["ultra_mensal", "ultra_anual", "pro_mensal", "pro_anual"]:
        # Pro — Claude Haiku > AIML(GPT-4o-mini) > Gemini
        if anthropic_ok:
            config["provedor"] = "claude_haiku"
            config["modelo_nome"] = "claude-haiku-4-5"
        elif aiml_ok:
            config["provedor"] = "aiml"
            config["modelo_nome"] = "gpt-4o-mini"
        else:
            config["provedor"] = "gemini"
            config["modelo_nome"] = "gemini-2.0-flash"
        config["max_tokens"] = 3000
        print(f"[CHAT] PRO — {config['modelo_nome']} / 3000 tokens")

    elif plano in ["creator_mensal", "creator_anual"]:
        # Creator — AIML(Gemini Pro) > Gemini
        if aiml_ok:
            config["provedor"] = "aiml"
            config["modelo_nome"] = "gemini-pro"
        else:
            config["provedor"] = "gemini"
            config["modelo_nome"] = "gemini-2.0-flash"
        config["max_tokens"] = 2500
        print(f"[CHAT] CREATOR — {config['modelo_nome']} / 2500 tokens")

    elif plano in ["starter_mensal", "starter_anual"]:
        config["provedor"] = "gemini"
        config["modelo_nome"] = "gemini-2.0-flash"
        config["max_tokens"] = 2000
        print(f"[CHAT] STARTER — Gemini / 2000 tokens")

    else:
        # Free — Gemini estável
        config["provedor"] = "gemini"
        config["modelo_nome"] = "gemini-2.0-flash"
        config["max_tokens"] = 1500
        print(f"[CHAT] FREE — Gemini / 1500 tokens")

    msgs = (data.historico or [])[-10:]
    
    # Injeta instrução de formato viral quando for pedido de roteiro
    from creditos import get_usuario as _get_user
    _plano_atual = _get_user(usuario_id).get("plano","free")
    msg_atual = data.texto.strip()
    if "Nova mensagem:" in msg_atual:
        msg_atual = msg_atual.split("Nova mensagem:")[-1].strip()
    elif "Histórico:" in msg_atual:
        linhas = [l.strip() for l in msg_atual.split("\n") if l.strip()]
        for l in reversed(linhas):
            if not l.startswith("Vortex:") and not l.startswith("Usuário:") and not l.startswith("Histórico:"):
                msg_atual = l
                break
    texto_final = msg_atual if msg_atual else data.texto

    # Tavily: busca fatos reais — usa APENAS a mensagem atual, não o histórico
    tavily_key = os.getenv("TAVILY_API_KEY", TAVILY_API_KEY)
    
    # Pega só a mensagem atual do usuário (sem histórico)
    msg_atual = data.texto.strip()
    
    # Remove prefixo de histórico se existir
    if "Nova mensagem:" in msg_atual:
        msg_atual = msg_atual.split("Nova mensagem:")[-1].strip()
    elif "Histórico:" in msg_atual:
        linhas = [l.strip() for l in msg_atual.split("\n") if l.strip()]
        # Pega última linha que não começa com "Vortex:" ou "Usuário:"
        for l in reversed(linhas):
            if not l.startswith("Vortex:") and not l.startswith("Usuário:") and not l.startswith("Histórico:"):
                msg_atual = l
                break
    
    # Palavras que ATIVAM busca — só casos/eventos reais específicos
    kw_busca_real = [
        "caso real","crime real","assassin","serial killer","true crime",
        "desaparec","acidente","tragédia","tragedia",
        "notícia","noticia","aconteceu em","o que aconteceu com",
        "quem foi","historia real","caso de",
    ]

    # Palavras que BLOQUEIAM busca — perguntas pessoais, sobre IA, opiniões
    kw_sem_busca = [
        # Perguntas sobre o Vortex/IA
        "você","vc","voce","você é","vc é","você sabe","vc sabe",
        "você pode","vc pode","você consegue","me ajuda","me ajude",
        "vortex","inteligência artificial","ia vai","substituir",
        "programador","desenvolvedor","claude","chatgpt","gemini","openai",
        # Perguntas pessoais / comparações
        "mais inteligente","quem é melhor","você ou","eu ou","qual melhor",
        "sua opinião","o que você acha","você prefere","você gosta",
        # Conceitos gerais que a IA já sabe
        "explica","como funciona","o que é","diferença entre","o que significa",
        "me ensina","me diz","me fala","me conta","me explica",
        # Perguntas casuais
        "oi","olá","ola","tudo bem","bom dia","boa tarde","boa noite",
        "obrigado","valeu","show","ótimo","legal","ok","certo",
        # Criação de conteúdo
        "roteiro","hook","viral","tiktok","reels","tendência","ideia",
        "cria","gera","escreve","faz um","me dá",
    ]

    msg_lower = msg_atual.lower()

    # Busca só é feita quando:
    # 1. Tem keyword de busca real
    # 2. Não tem keyword de bloqueio
    # 3. Mensagem tem mais de 20 chars (não é saudação)
    # 4. Não é resposta de histórico
    precisa_busca = (
        tavily_key and
        len(msg_atual) > 20 and
        any(kw in msg_lower for kw in kw_busca_real) and
        not any(kw in msg_lower for kw in kw_sem_busca) and
        "Histórico:" not in msg_atual[:20]
    )
    
    if precisa_busca:
        print(f"[Tavily] Buscando: {msg_atual[:60]}...")
        fatos = await buscar_tavily(msg_atual)
        if fatos:
            texto_final = "FATOS REAIS PESQUISADOS NA INTERNET:\n" + fatos + "\n\n---\nPEDIDO DO CRIADOR:\n" + data.texto + "\n\nUse os fatos reais acima. Seja específico — nomes reais, datas reais, locais reais, números reais."
            print(f"[Tavily] ✅ {len(fatos)} chars injetados")

    # Só acionar roteiro quando EXPLICITAMENTE pedido
    kw_roteiro = ["roteiro","script","gera roteiro","cria roteiro","fazer roteiro","monta roteiro","escreve roteiro"]
    eh_roteiro = any(kw in data.texto.lower() for kw in kw_roteiro)
    if eh_roteiro:
        anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY))
        if anthropic_ok:
            config["provedor"] = "claude_haiku"
            config["max_tokens"] = 3000
        else:
            config["provedor"] = "gemini"
            config["max_tokens"] = 2500
        print(f"[ROTEIRO] ✅ {config['provedor']} para roteiro com falas")
        # Reescrever o prompt do usuário forçando falas
        tema_roteiro = data.texto
        texto_final = f"""Crie um roteiro viral de 60 segundos sobre: {tema_roteiro}

ATENÇÃO: Você DEVE escrever as falas exatas do narrador em CADA cena. Sem falas = roteiro inútil.

Formato OBRIGATÓRIO:

🔥 TÍTULO VIRAL (3 opções)
1. "título aqui"
2. "título aqui"
3. "título aqui"

🎣 HOOK (0-3s)
NARRADOR: "frase perturbadora e específica — NÃO genérica"

🎬 ATO 1 — CONFLITO IMEDIATO (3-15s)
[0:03] NARRADOR: "primeira fala — o que o criador vai dizer"
[SOM: som específico real]
[0:08] NARRADOR: "segunda fala criando tensão"
[VISUAL: descrição rápida do visual]
[0:12] NARRADOR: "terceira fala"

🎬 ATO 2 — ESCALADA EMOCIONAL (15-40s)
[0:15] NARRADOR: "fala que aumenta o mistério"
[SOM: efeito]
[0:25] NARRADOR: "revelação perturbadora em palavras"
[VISUAL: cena]
[0:35] NARRADOR: "fala que deixa o espectador com frio na espinha"

🎬 ATO 3 — VIRADA IMPOSSÍVEL DE PREVER (40-50s)
[0:40] NARRADOR: "revelação chocante — fala direta e específica"
[SOM: efeito dramático]
[VISUAL: visual da virada]

🎬 ATO 4 — CLIFFHANGER (50-60s)
[0:52] NARRADOR: "fala final que corta a respiração"
[VISUAL: encerramento]

🎵 ÁUDIO VIRAL
Música: [nome REAL de música existente]
Efeitos: [sons específicos]

🖼️ PROMPT THUMBNAIL
[descrição visual objetiva para gerar imagem]

📱 LEGENDA
"legenda com até 150 chars"

#️⃣ HASHTAGS
#tag1 #tag2 [15 total]

📊 SCORE VIRAL (honesto — genérico = máximo 7/10)
🎣 Hook: X/10 | ⏱️ Retenção: X/10 | ❤️ Emoção: X/10 | 🔄 Shares: X/10 | 💬 Comentário: X/10
MÉDIA: X/10 — [VIRAL/POTENCIAL/RETRABALHAR]"""
        msgs = (data.historico or [])[-6:] + [{"role":"user","content":texto_final}]
    if False and eh_roteiro:
        texto_final = texto_final + """

CRIE O ROTEIRO EXATAMENTE NESTE FORMATO — com falas reais entre aspas:

🔥 TÍTULO VIRAL (3 opções)
1. "título 1"
2. "título 2"  
3. "título 3"

🎣 HOOK (0-3s)
NARRADOR: "frase perturbadora e específica do tema aqui"

🎬 ATO 1 — CONFLITO IMEDIATO (3-15s)
[0:03] NARRADOR: "fala exata que o criador vai dizer"
[SOM: descrição do efeito sonoro específico — ex: porta rangendo, passos pesados]
[0:08] NARRADOR: "segunda fala criando tensão"
[VISUAL: o que aparece na tela]

🎬 ATO 2 — ESCALADA EMOCIONAL (15-40s)
[0:15] NARRADOR: "fala que aumenta a tensão"
[SOM: efeito sonoro]
[0:25] NARRADOR: "fala revelando algo perturbador"
[VISUAL: descrição da cena]
[0:35] NARRADOR: "fala que deixa o espectador ansioso"

🎬 ATO 3 — VIRADA IMPOSSÍVEL DE PREVER (40-50s)
[0:40] NARRADOR: "revelação chocante em fala direta"
[SOM: efeito dramático]
[VISUAL: cena da virada]

🎬 ATO 4 — CLIFFHANGER (50-60s)
[0:52] NARRADOR: "fala final que corta a respiração"
[VISUAL: encerramento impactante]

🎵 ÁUDIO VIRAL
Música: [nome real de música que existe — ex: "Billie Eilish - bury a friend", "Hans Zimmer - Time"]
Efeitos: [lista de efeitos sonoros específicos]

🖼️ PROMPT THUMBNAIL
[descrição visual objetiva e detalhada para gerar a imagem]

📱 LEGENDA
"texto da legenda com até 150 caracteres"

#️⃣ HASHTAGS
#tag1 #tag2 #tag3 [15 no total]

📊 SCORE VIRAL (seja honesto — genérico não passa de 7)
🎣 Hook: X/10 | ⏱️ Retenção: X/10 | ❤️ Emoção: X/10 | 🔄 Shares: X/10 | 💬 Comentário: X/10
MÉDIA: X/10 — [VIRAL/POTENCIAL/RETRABALHAR]"""
    
    msgs.append({"role":"user","content":texto_final})
    # Retry automático — tenta até 2 vezes
    resposta, provedor = None, "desconhecido"
    try:
        # Usar cascata dedicada de chat (DeepSeek V3 → Qwen3 → Llama → Gemini)
        if config["provedor"] == "aiml":
            # AIML API — Claude, GPT-4o, Gemini Pro via uma key
            try:
                from providers import chamar_aiml
                modelo_aiml = config.get("modelo_nome", "gpt-4o")
                msgs_aiml = [{"role": m.get("role","user"), "content": m.get("content", m.get("text",""))} for m in msgs]
                resposta = await chamar_aiml(msgs_aiml, system=system, modelo=modelo_aiml, max_tokens=config["max_tokens"])
                provedor = f"aiml/{modelo_aiml}"
                print(f"[CHAT] ✅ AIML respondeu com {modelo_aiml}")
            except Exception as e_aiml:
                print(f"[CHAT] AIML falhou ({e_aiml}) — fallback Gemini")
                resposta, provedor = await gerar_texto(
                    messages=msgs, system=system,
                    max_tokens=config["max_tokens"],
                    provedor_preferido="gemini",
                )
        elif config["provedor"] == "openrouter_chat":
            resposta, provedor = await gerar_texto_chat(
                messages=msgs, system=system,
                max_tokens=config["max_tokens"],
            )
        else:
            # Planos pagos usam Claude/Gemini diretamente
            resposta, provedor = await gerar_texto(
                messages=msgs, system=system,
                max_tokens=config["max_tokens"],
                provedor_preferido=config["provedor"],
            )
    except Exception as e_chat:
        print(f"[VORTEX] ❌ Chat falhou: {e_chat} — fallback Gemini")
        try:
            resposta, provedor = await gerar_texto(
                messages=msgs, system=system,
                max_tokens=1000,
                provedor_preferido="gemini",
            )
        except:
            raise HTTPException(500, "Serviço temporariamente indisponível. Tente novamente.")
    debitar_creditos(usuario_id, 1, "chat")
    incrementar_limite_diario(usuario_id, "chat")
    incrementar_limite(usuario_id, "chat")
    lim["usado"] += 1

    # Adiciona nota de upgrade para usuários Free
    resposta_final = resposta
    if plano == "free" and any(kw in data.texto.lower() for kw in ["roteiro","video","viral","hook"]):
        resposta_final += """

---
⚡ **Quer roteiros ainda mais poderosos?**
O plano Pro usa Gemini 2.0 Flash e entrega roteiros 2x mais detalhados com direção cinematográfica completa. O Elite usa Claude Sonnet — nível Hollywood real.
👉 Upgrade em **Créditos** no menu lateral."""

    return {"ok":True,"resposta":resposta_final,"modelo_usado":provedor,
            "complexidade":complexidade.value,"cloudo_ativo":usar_cloudo,"limite_diario":lim}



# ══════════════════════════════════════════════════════════════
# ROTA — ROTEIRO VIRAL (Claude obrigatório — sem fallback)
# ══════════════════════════════════════════════════════════════

@app.post("/roteiro-viral")
async def roteiro_viral(request: Request):
    data = await request.json()
    tema = data.get("tema", "")
    nicho = data.get("nicho", "terror")
    
    ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    
    prompt = f"""Crie um roteiro viral de 60 segundos sobre: {tema}
Nicho: {nicho}

REGRA ABSOLUTA: Cada cena TEM que ter a fala exata do narrador entre aspas.
Sem fala = cena inválida.

🔥 TÍTULO VIRAL (3 opções)
1. "título criativo"
2. "título criativo"
3. "título criativo"

🎣 HOOK (0-3s)
NARRADOR: "frase perturbadora e específica sobre {tema}"

🎬 ATO 1 — CONFLITO IMEDIATO (3-15s)
[0:03] NARRADOR: "fala exata — o que o criador vai dizer em voz alta"
[SOM: som real e específico]
[0:08] NARRADOR: "segunda fala criando tensão imediata"
[VISUAL: descrição rápida]
[0:12] NARRADOR: "terceira fala"

🎬 ATO 2 — ESCALADA EMOCIONAL (15-40s)
[0:15] NARRADOR: "fala que aumenta o mistério"
[SOM: efeito sonoro]
[0:25] NARRADOR: "revelação perturbadora em palavras diretas"
[0:35] NARRADOR: "fala que deixa frio na espinha"

🎬 ATO 3 — VIRADA IMPOSSÍVEL (40-50s)
[0:40] NARRADOR: "revelação chocante — específica do tema"
[SOM: impacto sonoro]
[VISUAL: cena da virada]

🎬 ATO 4 — CLIFFHANGER (50-60s)
[0:52] NARRADOR: "fala final que corta a respiração"
[VISUAL: encerramento impactante]

🎵 ÁUDIO VIRAL
Música: [nome REAL de música que EXISTE no Spotify — ex: "Billie Eilish - bury a friend", "Hans Zimmer - Time", "Carpenter Brut - Turbo Killer". NUNCA invente músicas que não existem]
Efeitos: [sons específicos e reais]

🖼️ PROMPT THUMBNAIL
[descrição visual detalhada e objetiva]

📱 LEGENDA
"texto com até 150 chars"

#️⃣ HASHTAGS
#tag1 #tag2 #tag3 [15 total]

📊 SCORE VIRAL (seja honesto — tema genérico = máximo 7)
🎣 Hook: X/10 | ⏱️ Retenção: X/10 | ❤️ Emoção: X/10 | 🔄 Shares: X/10 | 💬 Comentário: X/10
MÉDIA: X/10 — [VIRAL/POTENCIAL/RETRABALHAR]"""

    # Buscar casos reais com Tavily para enriquecer o roteiro
    fatos_reais = ""
    if os.getenv("TAVILY_API_KEY", TAVILY_API_KEY):
        try:
            fatos_reais = await buscar_tavily(f"{tema} caso real Brasil {nicho}")
        except:
            pass  # silencioso intencional

    if fatos_reais:
        prompt = f"""DADOS REAIS PESQUISADOS (use como base):
{fatos_reais[:800]}

---
{prompt}

INSTRUÇÃO EXTRA: Use os dados reais acima. Seja hiper-específico — nomes, datas, lugares reais tornam o roteiro 4x mais viral."""

    system = montar_system_vortex(usar_cloudo=True)
    
    try:
        roteiro, modelo_usado = await gerar_texto_roteiro(
            [{"role":"user","content":prompt}],
            system=system,
            max_tokens=2000,
        )
        return {"ok": True, "roteiro": roteiro}
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar roteiro: {str(e)[:200]}")

# ══════════════════════════════════════════════════════════════
# ROTA — ROTEIRO (com todos os modos)
# ══════════════════════════════════════════════════════════════

@app.post("/gerar-roteiro")
@limiter.limit("10/minute")  # max 10 roteiros por minuto por IP
async def gerar_roteiro(data: RoteiroIn, request: Request):
    usuario_id = extrair_usuario_id(request, data)
    creditos_necessarios = {"curto":1,"medio":2,"longo":3,"completo":5}.get(data.formato, 2)
    if verificar_saldo(usuario_id, creditos_necessarios) < creditos_necessarios:
        raise HTTPException(402, "Créditos insuficientes.")
    
    # Verificar limite de roteiros do plano free
    lim = checar_limite(usuario_id)
    if lim["is_free"] and lim["roteiros_hoje"] >= lim["limite_roteiros"]:
        raise HTTPException(429, f"Você usou os {lim['limite_roteiros']} roteiros grátis de hoje. Faça upgrade para roteiros ilimitados! 🚀")
    
    # Incrementar contador de roteiros — persiste no arquivo
    incrementar_limite(usuario_id, "roteiro")
    _limite["roteiros_hoje"] = _limite.get("roteiros_hoje", 0) + 1

    _perfil = carregar_perfil(data.canal_id or "default") or {}
    plataforma = _perfil.get("plataformas", ["TikTok"])[0] if _perfil.get("plataformas") else "TikTok"
    nicho      = data.nicho or _perfil.get("nicho", "conteúdo viral")
    tom        = _perfil.get("tom_de_voz", "direto e impactante")
    publico    = _perfil.get("publico_alvo", "criadores de conteúdo")
    nome_canal = _perfil.get("nome_canal", "")
    contexto   = montar_contexto_criador(data.canal_id or "default")

    MAX_TOKENS = {"curto":1200,"medio":2000,"longo":3000,"completo":4000}.get(data.formato, 2000)

    # Buscar fatos reais e tendências em paralelo
    fatos_reais = ""
    tendencias_reais = ""
    if TAVILY_API_KEY:
        try:
            import asyncio
            fatos_task    = buscar_tavily(f"{data.tema} caso real Brasil viral")
            tends_task    = buscar_tavily(f"tendências {nicho} {plataforma} viral 2026")
            fatos_reais, tendencias_reais = await asyncio.gather(fatos_task, tends_task, return_exceptions=True)
            if isinstance(fatos_reais, Exception):    fatos_reais = ""
            if isinstance(tendencias_reais, Exception): tendencias_reais = ""
        except:
            pass  # silencioso intencional

    system = """Você é o VORTEX SCRIPT ENGINE — o melhor roteirista de conteúdo viral do Brasil.
Você já criou roteiros que geraram mais de 50 milhões de views no TikTok e Instagram.

FILOSOFIA:
- Hook nos primeiros 2 segundos = vida ou morte do vídeo
- Especificidade vende — "homem de 34 anos em São Paulo" > "uma pessoa"
- Emoção > informação sempre
- Cada segundo deve ter uma função: prender, revelar, ou chocar
- O final deve ser impossível de não compartilhar

REGRAS ABSOLUTAS:
- NUNCA começa com pergunta
- NUNCA usa "Olá", "Hoje vamos falar" ou "Você sabia que"
- NUNCA entrega roteiro com score < 8/10 — reescreve internamente
- SEMPRE entrega falas 100% completas — zero "[falar sobre X]"
- SEMPRE em português brasileiro natural e fluido
- SEMPRE com score viral real ao final"""

    # Contexto do criador
    if contexto:
        system += "\n\nPERFIL DO CRIADOR:\n" + contexto + "\nPersonalize tudo ao perfil acima."





    # ── MODO NORMAL — roteiro viral padrão ──────────────────────────
    if data.modo in ["normal", "viral", ""]:
        duracao = {"curto":"30-45s","medio":"60s","longo":"2-3min","completo":"3-5min"}.get(data.formato,"60s")
        
        prompt = f"""Crie um roteiro viral PROFISSIONAL de {duracao} para {plataforma}.

BRIEFING:
• Tema: {data.tema}
• Nicho: {nicho}
• Tom: {tom}
• Público: {publico}
• Plataforma: {plataforma}
{f"• Canal: {nome_canal}" if nome_canal else ""}
{f"• Tendências reais do nicho:{chr(10)}{tendencias_reais[:400]}" if tendencias_reais else ""}
{f"• Fatos reais pesquisados:{chr(10)}{fatos_reais[:500]}" if fatos_reais else ""}

ENTREGUE EXATAMENTE NESTE FORMATO:

╔══════════════════════════════════════╗
║  🎬 ROTEIRO — {data.tema}
╚══════════════════════════════════════╝

⚡ HOOK (0-3s):
[Frase de abertura que PARA o scroll — afirmação chocante, número impossível ou revelação]

🔥 DESENVOLVIMENTO:
[Roteiro completo com todas as falas, cenas e virada inesperada]

💥 CLÍMAX + CTA:
[Final que força compartilhamento, comentário ou salvar]

🎵 PRODUÇÃO:
• Música: [gênero + segundo do beat drop]
• Thumbnail: [descrição visual cinematográfica]
• Legenda: [texto completo com emojis]
• Hashtags: #[15 hashtags estratégicas separadas]

📊 SCORE VIRAL:
• Hook: X/10 | Retenção: X/10 | Emoção: X/10 | Shares: X/10 | Comentários: X/10
• MÉDIA: X/10
• POTENCIAL: [estimativa de views]
• MELHOR HORÁRIO PARA POSTAR: [dia da semana, hora]"""

    # ── MODO DIRETOR — segundo a segundo ──────────────────────────
    elif data.modo == "diretor":
        prompt = f"""Crie um roteiro DIRETOR COMPLETO — frame por frame — para: {data.tema}

Plataforma: {plataforma} | Nicho: {nicho}
{f"Fatos reais:{chr(10)}{fatos_reais[:600]}" if fatos_reais else ""}

ENTREGUE:
[00:00-00:03] HOOK:
• CÂMERA: [ângulo exato + movimento]
• CENA: [o que aparece na tela]
• FALA: "[texto completo do narrador]"
• EMOÇÃO DO ESPECTADOR: [o que sente]
• SOM: [música + efeito sonoro]

[Continue para cada cena até o final]

TRANSIÇÕES: [tipo de corte entre cada cena]
EFEITOS: [filtros, texto na tela, emojis]

📊 SCORE: Hook X/10 | Retenção X/10 | Emoção X/10 | Shares X/10 | Comentário X/10 | MÉDIA: X/10"""

    # ── MODO SÉRIE — 3 episódios ──────────────────────────────────
    elif data.modo == "serie":
        prompt = f"""Crie uma SÉRIE VIRAL de 3 episódios sobre: {data.tema}
Plataforma: {plataforma} | Nicho: {nicho}

Para cada episódio:
━━ EPISÓDIO [N] ━━
• Hook individual: [frase que funciona SOZINHA]
• Roteiro completo: [todas as falas]
• Cliffhanger: [por que o espectador VÁ ver o próximo]
• Hashtag da série: #[nome da série]
• Score: X/10

Faça os 3 episódios completos. Cada um deve funcionar sozinho E criar desejo pelo próximo."""

    # ── MODO A/B — 2 versões ─────────────────────────────────────
    elif data.modo == "ab":
        prompt = f"""Crie 2 VERSÕES do mesmo roteiro sobre: {data.tema}
Plataforma: {plataforma} | Nicho: {nicho}

VERSÃO A — Hook emocional (apela para sentimento):
[Roteiro completo]
Score A: X/10

VERSÃO B — Hook chocante (apela para curiosidade):
[Roteiro completo]  
Score B: X/10

VEREDICTO: Qual versão vai melhor e por quê em 2 linhas."""

    # ── MODO FACELESS — sem aparecer ─────────────────────────────
    elif data.modo == "faceless":
        prompt = f"""Crie um roteiro FACELESS COMPLETO — sem mostrar rosto — sobre: {data.tema}
Plataforma: {plataforma} | Nicho: {nicho}

ESTRUTURA FACELESS:
• Narração em off: [texto completo para narrar]
• Imagens sugeridas: [o que mostrar em cada trecho]
• Texto na tela: [frases que aparecem sobrepostas]
• Música: [gênero e energia]
• Prompt de imagem IA: [prompt para gerar a thumbnail]

Score: X/10 | Dificuldade de produção: [fácil/médio/difícil]"""

    # ── MODO ANÚNCIO — conversão ──────────────────────────────────
    elif data.modo == "anuncio":
        prompt = f"""Crie um roteiro de ANÚNCIO que converte sobre: {data.tema}
Plataforma: {plataforma} | Nicho: {nicho}

ESTRUTURA: Hook (3s) → Problema (10s) → Solução (20s) → Prova (15s) → CTA (5s)

[Roteiro completo com todas as falas]

• Gatilhos de conversão usados: [lista]
• CTA exato: [a frase exata para o call-to-action]
• Taxa de conversão esperada: [estimativa]"""

    else:
        # Fallback — modo normal
        prompt = f"Crie um roteiro viral profissional sobre: {data.tema} para {plataforma}. Nicho: {nicho}. Entregue hook, desenvolvimento, CTA e score viral."

    # Injetar fatos reais no prompt se disponível e não foi usado ainda
    if fatos_reais and "fatos reais" not in prompt.lower()[:100]:
        prompt = "FATOS REAIS PESQUISADOS (use como base):\n" + fatos_reais[:800] + "\n\n---\n" + prompt





    # Cascata de modelos — DeepSeek > Groq > Gemini
    roteiro_texto, modelo_usado = await gerar_texto_roteiro(
        [{"role": "user", "content": prompt}],
        system=system,
        max_tokens=MAX_TOKENS,
    )

    debitar_creditos(usuario_id, creditos_necessarios, f"roteiro_{data.formato}_{data.modo}")

    # Detectar qualidade do modelo usado no roteiro
    MODELOS_TOP_ROT = ["claude", "gpt-4", "deepseek-v3", "qwen3-235b", "llama-3.1-405b"]
    modelo_lower = modelo_usado.lower()
    qualidade_rot = "top" if any(m in modelo_lower for m in MODELOS_TOP_ROT) else "medio"

    aviso_rot = None
    if qualidade_rot != "top":
        aviso_rot = {
            "tipo": "qualidade_reduzida",
            "titulo": "⚡ IAs premium no limite",
            "mensagem": f"O modelo usado foi {modelo_usado}. Para roteiros com DeepSeek V3 ou Claude garantidos, adicione créditos.",
            "btn_label": "💎 Recarregar para garantir IA top",
            "btn_aba": "creditos",
        }

    return {
        "ok": True,
        "roteiro": roteiro_texto,
        "tema": data.tema,
        "formato": data.formato,
        "modo": data.modo,
        "modelo": modelo_usado,
        "qualidade": qualidade_rot,
        "aviso": aviso_rot,
        "creditos_debitados": creditos_necessarios,
        "plataforma": plataforma,
        "nicho": nicho,
    }

@app.post("/score-viral")
async def score_viral(data: ScoreRequest, request: Request):
    usuario_id = extrair_usuario_id(request, data)
    # Score Viral é GRÁTIS e ILIMITADO — feature de vício do Vortex
    # Não cobra créditos, não tem limite por plano

    nicho = data.nicho or _perfil.get("nicho","conteúdo digital")
    prompt = f"""Analise este roteiro e dê um SCORE VIRAL detalhado:

ROTEIRO:
{data.roteiro[:2000]}

NICHO: {nicho}

Avalie em 5 dimensões (0-10 cada):
🎣 HOOK: O hook para o scroll nos primeiros 2 segundos?
⏱️ RETENÇÃO: Cada segundo justifica o próximo?
❤️ EMOÇÃO: Gera sentimento físico no espectador?
🔄 COMPARTILHAMENTO: O espectador vai enviar para alguém?
💬 COMENTÁRIO: Provoca resposta espontânea?

Para cada dimensão:
- Nota (0-10)
- O que está funcionando
- O que melhorar
- Exemplo de como melhorar

M�dia final e veredicto: VIRAL / POTENCIAL / RETRABALHAR"""

    resultado, _ = await gerar_texto(
        [{"role":"user","content":prompt}],
        system="Você é um analista de conteúdo viral sênior. Seja preciso e direto.",
        max_tokens=1200, provedor_preferido="groq",
    )
    debitar_creditos(usuario_id, 2, "score_viral")
    return {"ok":True,"score":resultado,"nicho":nicho,"creditos_debitados":2}


# ══════════════════════════════════════════════════════════════
# ROTA — DNA DO CRIADOR
# ══════════════════════════════════════════════════════════════

@app.post("/dna-criador")
async def aprender_dna(data: DNARequest, req: Request):
    usuario_id = extrair_usuario_id(req)
    saldo = verificar_saldo(usuario_id, 3)
    if saldo < 3: raise HTTPException(402, "Créditos insuficientes. Precisa de 3.")

    roteiros_txt = "\n\n---\n\n".join(data.roteiros_aprovados[:5])
    prompt = f"""Analise estes roteiros que viralizaram para este criador e identifique o DNA único dele:

{roteiros_txt}

{f"O criador descreve seu estilo como: {data.estilo_descricao}" if data.estilo_descricao else ""}

Identifique:
1. Padrão de hook (como ele começa os vídeos)
2. Ritmo da narrativa (rápido/lento/variado)
3. Vocabulário e gírias recorrentes
4. Tipo de emoção que mais usa
5. Estrutura de CTA preferida
6. 3 características únicas do estilo dele
7. Fórmula secreta do sucesso dele em 1 frase"""

    analise, _ = await gerar_texto(
        [{"role":"user","content":prompt}],
        system="Você é um analista de estilo de conteúdo. Seja específico e preciso.",
        max_tokens=800, provedor_preferido="groq",
    )

    # Salva DNA
    _dna_criador.update({
        "analise": analise,
        "roteiros_aprovados": data.roteiros_aprovados,
        "estilo_predominante": data.estilo_descricao or "analisado automaticamente",
        "hooks_aprovados": [r[:100] for r in data.roteiros_aprovados],
        "atualizado_em": str(datetime.now()),
    })
    _system_cache["hash"] = ""
    salvar_dna(_dna_criador)
    debitar_creditos(usuario_id, 3, "dna_criador")
    return {"ok":True,"dna":analise,"salvo":True,"creditos_debitados":3}


# ══════════════════════════════════════════════════════════════
# ROTA — MODO AGÊNCIA (múltiplos canais)
# ══════════════════════════════════════════════════════════════

@app.post("/canais")
def criar_canal(data: CanalRequest):
    _canais[data.canal_id] = data.dict()
    salvar_canais(_canais)
    return {"ok":True,"canal_id":data.canal_id,"total_canais":len(_canais)}

@app.get("/canais")
def listar_canais():
    return {"ok":True,"canais":list(_canais.keys()),"total":len(_canais)}

@app.get("/canais/{canal_id}")
def get_canal(canal_id: str):
    if canal_id not in _canais: raise HTTPException(404, "Canal não encontrado")
    return {"ok":True,"canal":_canais[canal_id]}


# ══════════════════════════════════════════════════════════════
# ROTA — CALENDÁRIO INTELIGENTE
# ══════════════════════════════════════════════════════════════

@app.get("/calendario")
async def calendario(req: Request):
    usuario_id = extrair_usuario_id(req)
    saldo = verificar_saldo(usuario_id, 2)
    if saldo < 2: raise HTTPException(402, "Créditos insuficientes.")

    nicho = _perfil.get("nicho","lifestyle")
    plataformas = _perfil.get("plataformas",["TikTok"])
    plat = plataformas[0] if isinstance(plataformas,list) else plataformas

    horarios = MELHORES_HORARIOS.get(plat, MELHORES_HORARIOS["TikTok"])
    tendencias_nicho = TENDENCIAS_2026.get(nicho.lower(), TENDENCIAS_2026.get("lifestyle",{}))
    trends_plat = tendencias_nicho.get(plat, [])

    prompt = f"""Crie um CALENDÁRIO DE CONTEÚDO para a próxima semana:

Nicho: {nicho}
Plataforma: {plat}
Horários ideais: {json.dumps(horarios)}
Tendências do nicho: {', '.join(trends_plat[:5])}

Para cada dia da semana (seg a dom), sugira:
- Tipo de conteúdo ideal para aquele dia
- Tema específico baseado nas tendências
- Horário exato para postar
- Formato (reels 30s / vídeo longo / carrossel)
- Hook sugerido para o vídeo"""

    calendario_txt, _ = await gerar_texto(
        [{"role":"user","content":prompt}],
        system=montar_system_vortex(usar_cloudo=True),
        max_tokens=1200, provedor_preferido="groq",
    )
    debitar_creditos(usuario_id, 2, "calendario")
    return {"ok":True,"calendario":calendario_txt,"plataforma":plat,"nicho":nicho,
            "horarios_ideais":horarios,"creditos_debitados":2}


# ══════════════════════════════════════════════════════════════
# ROTA — RELATÓRIO SEMANAL
# ══════════════════════════════════════════════════════════════

@app.get("/relatorio-semanal")
async def relatorio_semanal(req: Request):
    usuario_id = extrair_usuario_id(req)
    saldo = verificar_saldo(usuario_id, 3)
    if saldo < 3: raise HTTPException(402, "Créditos insuficientes.")

    nicho = _perfil.get("nicho","conteúdo digital")
    plataformas = _perfil.get("plataformas",["TikTok"])
    hist = historico_creditos(usuario_id)
    acoes = [h.get("acao","") for h in hist[-20:]] if hist else []

    prompt = f"""Crie um RELATÓRIO SEMANAL de criação de conteúdo:

Nicho: {nicho}
Plataformas: {plataformas}
Ações desta semana: {', '.join(acoes) if acoes else 'Primeira semana'}

O relatório deve incluir:
1. 📊 RESUMO DA SEMANA — o que foi produzido
2. 🔥 TENDÊNCIAS QUENTES — o que está viralizando agora no nicho
3. 💡 3 IDEIAS PARA PRÓXIMA SEMANA — baseadas nas tendências
4. ⚠️ O QUE EVITAR — erros comuns no nicho agora
5. 🎯 META DA SEMANA — um objetivo específico e mensurável
6. 🚀 DICA DE OURO — uma insight que poucos criadores sabem"""

    relatorio, _ = await gerar_texto(
        [{"role":"user","content":prompt}],
        system=montar_system_vortex(usar_cloudo=True),
        max_tokens=1500, provedor_preferido="groq",
    )
    debitar_creditos(usuario_id, 3, "relatorio_semanal")
    return {"ok":True,"relatorio":relatorio,"semana":str(date.today()),"creditos_debitados":3}


# ══════════════════════════════════════════════════════════════
# ROTA — SUGESTÃO PROATIVA DE TRENDS
# ══════════════════════════════════════════════════════════════

@app.get("/trends-agora")
async def trends_agora(request: Request):
    nicho = _perfil.get("nicho","lifestyle")
    plataformas = _perfil.get("plataformas",["TikTok"])
    plat = plataformas[0] if isinstance(plataformas,list) else plataformas
    
    # Busca trends reais via Tavily
    fatos_reais = ""
    tavily_key = os.getenv("TAVILY_API_KEY", TAVILY_API_KEY)
    if tavily_key:
        query = f"tendências virais {nicho} {plat} Brasil 2026 maio"
        fatos_reais = await buscar_tavily(query, max_results=3)
        print(f"[Trends] Tavily buscou: {len(fatos_reais)} chars")

    tendencias_nicho = TENDENCIAS_2026.get(nicho.lower(), {})
    trends_base = tendencias_nicho.get(plat, ["Conteúdo autêntico","Consistência","Engajamento real"])

    prompt = f"""Você é o maior especialista em tendências de conteúdo viral do Brasil em 2026.

NICHO: {nicho}
PLATAFORMA: {plat}
DATA: {datetime.now().strftime('%d/%m/%Y')}

{dados_str}

Com base nesses dados REAIS e atuais, entregue:

🔥 TOP 5 TRENDS AGORA
Para cada trend:
- Nome da tendência
- Por que está viralizando AGORA (dados reais se possível)
- Como adaptar para {nicho} no {plat}
- Ideia de vídeo pronto para gravar HOJE
- Urgência: 🔴 USE AGORA / 🟡 Esta semana / 🟢 Este mês

⚡ DICA DE OURO
Uma insight que poucos criadores de {nicho} sabem sobre o algoritmo do {plat} agora."""

    sugestao, _ = await gerar_texto(
        [{"role":"user","content":prompt}],
        system=montar_system_vortex(usar_cloudo=True),
        max_tokens=1200, provedor_preferido="groq",
    )
    return {"ok":True,"trends":sugestao,"nicho":nicho,"plataforma":plat,
            "trends_raw":trends_base,"fatos_reais":bool(fatos_reais),"atualizado_em":str(datetime.now())}


# ══════════════════════════════════════════════════════════════
# ROTA — ANÁLISE DE PERFIL
# ══════════════════════════════════════════════════════════════

@app.post("/analisar-perfil")
async def analisar_perfil(data: AnalisarPerfilIn, req: Request):
    usuario_id = extrair_usuario_id(req)
    saldo = verificar_saldo(usuario_id, 2)
    if saldo < 2: raise HTTPException(402, "Créditos insuficientes.")
    rede = data.rede.lower()
    perfil = data.perfil.strip().lstrip("@").split("/")[-1]
    if not perfil: raise HTTPException(400, "Perfil não informado.")
    handlers = {"instagram":analisar_instagram,"tiktok":analisar_tiktok,"youtube":analisar_youtube}
    handler = handlers.get(rede)
    if not handler: raise HTTPException(400, f"Rede '{rede}' não suportada.")
    try:
        resultado = await handler(perfil)
    except Exception as e:
        raise HTTPException(502, f"Erro ao buscar dados: {str(e)}")
    if GROQ_API_KEY or GEMINI_API_KEY:
        try:
            contexto = montar_contexto_criador()
            
            # Buscar dados reais do nicho com Tavily
            dados_mercado = ""
            if TAVILY_API_KEY:
                try:
                    nicho_perfil = resultado.get("nicho", "") or resultado.get("bio", "")[:50]
                    dados_mercado = await buscar_tavily(f"estratégia crescimento {rede} {nicho_perfil} 2026 Brasil")
                except:
                    pass  # silencioso intencional

            # Score de engajamento calculado
            seguidores = resultado.get('seguidores', 0) or 0
            eng_raw = resultado.get('engajamento', '0%')
            posts = resultado.get('posts', 0) or 0
            bio = resultado.get('bio', '') or ''
            
            prompt_ia = f"""Você é o estrategista de {rede} mais preciso e cirúrgico do Brasil.
Analise o perfil @{perfil} com dados reais e entregue diagnóstico de alto valor.

═══════════════════════════════════
DADOS REAIS DO PERFIL @{perfil}
═══════════════════════════════════
• Seguidores: {seguidores:,}
• Taxa de engajamento: {eng_raw}
• Total de posts: {posts}
• Bio: {bio[:150] if bio else "Não disponível"}
• Rede: {rede.upper()}
{f"• Contexto adicional: {contexto[:200]}" if contexto else ""}
{f"• Benchmark do mercado: {dados_mercado[:500]}" if dados_mercado else ""}

═══════════════════════════════════
ENTREGUE OBRIGATORIAMENTE:
═══════════════════════════════════

🎯 SCORE DO PERFIL: X/100
• Consistência: X/10 — [diagnóstico em 1 linha]
• Engajamento: X/10 — [diagnóstico em 1 linha]
• Posicionamento de nicho: X/10 — [diagnóstico em 1 linha]
• Potencial viral: X/10 — [diagnóstico em 1 linha]
• Bio/Primeira impressão: X/10 — [diagnóstico em 1 linha]

📊 DIAGNÓSTICO REAL
O que está FUNCIONANDO: [específico, não genérico]
O que está MATANDO o crescimento: [específico, com dados]

⚠️ ERRO #1 (o maior freio do perfil agora):
[Nome do erro] — [por que isso mata o crescimento e como provar com os dados]

🚀 3 AÇÕES DOS PRÓXIMOS 7 DIAS:
1. [Ação específica e mensurável — com prazo e métrica]
2. [Ação específica e mensurável — com prazo e métrica]  
3. [Ação específica e mensurável — com prazo e métrica]

💡 INSIGHT QUE NINGUÉM VÊ:
[Uma oportunidade não óbvia específica para esse perfil com esses dados]

🔮 PREVISÃO:
Se implementar as 3 ações em 30 dias: [estimativa realista de crescimento %]"""

            recomendacao, _ = await gerar_texto_chat(
                [{"role":"user","content":prompt_ia}],
                system="Você é o estrategista de redes sociais mais preciso do Brasil. Seja específico, direto e cirúrgico.",
                max_tokens=800,
            )
            resultado["analise_completa"] = recomendacao
        except Exception as e:
            print(f"[Análise IA] {e}")
    debitar_creditos(usuario_id, 2, "analise_perfil")
    return {"ok":True,"rede":rede,"perfil":perfil,**resultado}




# ══════════════════════════════════════════════════════════════════
# 🎬 ANÁLISE DE VÍDEO — Por URL ou upload
# ══════════════════════════════════════════════════════════════════
class AnalisarVideoIn(BaseModel):
    url: Optional[str] = ""
    transcricao: Optional[str] = ""
    titulo: Optional[str] = ""
    views: Optional[int] = 0
    likes: Optional[int] = 0
    comentarios: Optional[int] = 0
    nicho: Optional[str] = "viral"

@app.post("/analisar-video")
async def analisar_video(data: AnalisarVideoIn, request: Request):
    """
    Analisa um vídeo e retorna:
    - Por que está ou não viralizando
    - Score de retenção estimado
    - O que melhorar
    - Como replicar o sucesso
    """
    # Calcular métricas
    taxa_engajamento = 0
    if data.views and data.views > 0:
        taxa_engajamento = round(((data.likes or 0) + (data.comentarios or 0)) / data.views * 100, 2)

    # Buscar dados sobre o nicho
    dados_nicho = ""
    if TAVILY_API_KEY and data.nicho:
        try:
            dados_nicho = await buscar_tavily(f"vídeos virais {data.nicho} TikTok 2026 o que funciona")
        except:
            pass  # silencioso intencional

    prompt = f"""Você é o melhor analista de vídeos virais do Brasil.

DADOS DO VÍDEO:
- Título: {data.titulo or "Não informado"}
- Views: {data.views:,} 
- Likes: {data.likes:,}
- Comentários: {data.comentarios:,}
- Taxa de engajamento: {taxa_engajamento}%
- Nicho: {data.nicho}
{f"- Transcrição/Roteiro: {data.transcricao[:500]}" if data.transcricao else ""}
{f"- Dados do mercado: {dados_nicho[:300]}" if dados_nicho else ""}

Entregue uma análise CINEMATOGRÁFICA e CIRÚRGICA:

🎯 DIAGNÓSTICO VIRAL
Por que este vídeo está/não está viralizando (seja específico)

📊 SCORE DE RETENÇÃO: X/10
- Hook (0-3s): X/10 — [análise]
- Conflito (3-15s): X/10 — [análise]
- Escalada (15-40s): X/10 — [análise]
- Virada (40-50s): X/10 — [análise]
- Final (50-60s): X/10 — [análise]

💡 FÓRMULA DO SUCESSO
O que exatamente está fazendo este vídeo funcionar (ou não)

🚀 COMO REPLICAR
3 ideias concretas de vídeos baseados nesta análise

⚡ MELHORIA IMEDIATA
Se fosse refazer este vídeo, o que mudaria nos primeiros 3 segundos"""

    try:
        analise, _ = await gerar_texto_chat(
            [{"role": "user", "content": prompt}],
            system="Você é o analista de vídeos virais mais preciso do Brasil. Use dados reais e seja específico.",
            max_tokens=1000,
        )
        return {
            "ok": True,
            "metricas": {
                "views": data.views,
                "likes": data.likes,
                "comentarios": data.comentarios,
                "taxa_engajamento": taxa_engajamento,
                "classificacao": "VIRAL" if taxa_engajamento > 5 else "POTENCIAL" if taxa_engajamento > 2 else "BAIXO ENGAJAMENTO"
            },
            "analise": analise
        }
    except Exception as e:
        raise HTTPException(500, f"Erro na análise: {str(e)[:100]}")


# ══════════════════════════════════════════════════════════════
# ROTA — GERAR IMAGEM
# ══════════════════════════════════════════════════════════════

# ── Tradução PT→EN automática para APIs de IA ────────────────────────────────
async def traduzir_prompt(texto: str) -> str:
    palavras_pt = ["uma","um","de","do","da","com","para","em","no","na","que","se","por","como","um","jovem","mulher","homem","cena","correndo","andando","floresta","escuro","noite"]
    palavras = texto.lower().split()
    if not any(p in palavras for p in palavras_pt):
        return texto  # já em inglês
    try:
        instrucao = "Translate this to English for AI image/video generation. Return ONLY the translation: " + texto
        msgs = [{"role":"user","content":instrucao}]
        traduzido, _ = await gerar_texto(msgs, system="Translate to English only. No explanations.", max_tokens=150, provedor_preferido="groq")
        print(f"[PT→EN] {texto[:40]} → {traduzido[:40]}")
        return traduzido.strip()
    except:
        return texto

@app.post("/gerar-imagem")
async def gerar_imagem(request: ImageRequest, req: Request):
    usuario_id = extrair_usuario_id(req)
    # Créditos de imagem calculados com margem real (custo_fal × 1.4 / valor_credito_medio)
    creditos_map = {
        # FAL — custo real com margem 40%
        "flux-dev":              8,   # $0.025/img → 8cr
        "flux-schnell":          3,   # $0.003/img → 3cr
        "ideogram":              15,  # $0.08/img → 15cr
        "stability":             12,  # $0.065/img → 12cr
        # Grátis
        "pollinations":          0,
        "hf_flux":               0,
        "gemini":                0,
        "raphael":               0,
        # AIML
        "aiml_flux":             1,
        "aiml_flux_dev":         2,
        "aiml_gpt":              4,
        # Legado
        "wavespeed-ai/flux-dev": 8,
        "wavespeed-ai/flux-dev-ultra-fast": 5,
        "wavespeed-ai/flux-schnell": 3,
        "PHOENIX": 6,
        "LEONARDO_SIGNATURE": 10,
        "stability-ultra": 12,
        "ideogram-v2": 15,
    }
    creditos = creditos_map.get(request.modelo or "pollinations", 5)
    saldo = verificar_saldo(usuario_id, creditos)
    if saldo < creditos: raise HTTPException(402, f"Créditos insuficientes. Precisa de {creditos}.")

    # Sistema inteligente — entende qualquer idioma e otimiza para a API
    tipo_prompt = "thumbnail" if any(w in request.prompt.lower() for w in ["thumbnail","capa","miniatura","tiktok","youtube","reel","viral","click"]) else "imagem"
    
    # Estilo do request enriquece o contexto
    estilo_ctx = ""
    ESTILOS_MAP = {
        "realista": "photorealistic, professional photography, natural lighting, ",
        "cinematico": "cinematic shot, movie scene, dramatic lighting, anamorphic lens, ",
        "anime": "anime style, vibrant illustration, manga art, Studio Ghibli inspired, ",
        "dark": "dark atmosphere, moody, gothic, chiaroscuro lighting, sinister, ",
        "cartoon": "cartoon style, bold outlines, flat colors, comic book, ",
        "3d": "3D render, octane render, volumetric lighting, subsurface scattering, ",
    }
    if hasattr(request, 'estilo') and request.estilo and request.estilo in ESTILOS_MAP:
        estilo_ctx = ESTILOS_MAP[request.estilo]
    
    prompt_base = estilo_ctx + request.prompt if estilo_ctx else request.prompt
    prompt_en = prompt_base + ", highly detailed, 8k uhd, cinematic lighting, award winning"

    modelo_req = request.modelo or "wavespeed-ai/flux-dev"
    url = None
    modelo_usado = "desconhecido"
    erros = []

    # Cascata: Leonardo → WaveSpeed → erro claro
    if "stability" in modelo_req.lower() or "sdxl" in modelo_req.lower():
        url = await gerar_imagem_stability(prompt=prompt_en, width=request.width, height=request.height)
        modelo_usado = "Stability AI Ultra"
    elif "ideogram" in modelo_req.lower():
        url = await gerar_imagem_ideogram(prompt=prompt_en, width=request.width, height=request.height)
        modelo_usado = "Ideogram v2"
    elif "PHOENIX" in modelo_req or "LEONARDO" in modelo_req or "SIGNATURE" in modelo_req:
        if LEONARDO_API_KEY:
            try:
                url = await gerar_imagem_leonardo(prompt=prompt_en, width=request.width, height=request.height, modelo=modelo_req)
                modelo_usado = "Leonardo AI"
            except Exception as e:
                erros.append(f"Leonardo: {e}")
        if not url and WAVESPEED_API_KEY:
            try:
                url = await gerar_imagem_wavespeed(prompt=prompt_en, endpoint="wavespeed-ai/flux-dev")
                modelo_usado = "WaveSpeed Flux (fallback)"
            except Exception as e:
                erros.append(f"WaveSpeed fallback: {e}")
    else:
        # WaveSpeed primeiro → Leonardo fallback
        if WAVESPEED_API_KEY:
            try:
                url = await gerar_imagem_wavespeed(prompt=prompt_en, endpoint=modelo_req)
                modelo_usado = "WaveSpeed Flux"
            except Exception as e:
                erros.append(f"WaveSpeed: {e}")
        if not url and LEONARDO_API_KEY:
            try:
                url = await gerar_imagem_leonardo(prompt=prompt_en, width=request.width, height=request.height, modelo="PHOENIX")
                modelo_usado = "Leonardo AI (fallback)"
            except Exception as e:
                erros.append(f"Leonardo fallback: {e}")

    # FAL.ai — Flux Dev de alta qualidade
    if not url and FAL_API_KEY:
        try:
            url = await gerar_imagem_fal(prompt=prompt_en, modelo="fal-ai/flux/dev", width=request.width or 1024, height=request.height or 1024)
            modelo_usado = "FAL Flux Dev"
        except Exception as e:
            erros.append(f"FAL: {e}")

    # Fallback gratuito — Pollinations.ai (sem key, sempre disponível)
    if not url:
        try:
            import urllib.parse
            prompt_encoded = urllib.parse.quote(prompt_en[:500])
            url = f"https://image.pollinations.ai/prompt/{prompt_encoded}?width=1024&height=1024&model=flux&nologo=true&seed={hash(prompt_en) % 99999}"
            modelo_usado = "Pollinations Flux (free)"
            print(f"[Pollinations] URL gerada: {url[:80]}")
        except Exception as e:
            erros.append(f"Pollinations: {e}")

    if not url:
        raise HTTPException(500, f"Todas as APIs de imagem falharam: {'; '.join(erros)}")

    debitar_creditos(usuario_id, creditos, "gerar_imagem")
    return {"ok":True,"imagem":url,"modelo":modelo_usado,"prompt_en":prompt_en,"creditos_debitados":creditos}


# ══════════════════════════════════════════════════════════════
# ROTA — GERAR VÍDEO
# ══════════════════════════════════════════════════════════════

@app.post("/gerar-video")
async def gerar_video(request: VideoRequest, req: Request):
    usuario_id = extrair_usuario_id(req)
    creditos_base_map = {
        "wavespeed-ai/wan-2.2/t2v-480p": 6,
        "wavespeed-ai/wan-2.2/t2v-5b-720p": 12,
        "wavespeed-ai/wan-2.2/t2v-720p-ultra-fast": 10,
        "kling-v1": 15,
        "kling-v1-5": 20,
        "kling-v2": 25,
        "runway-gen3": 30,
        "runway-gen3-turbo": 20,
        "luma-dream-machine": 18,
        "luma-ray2": 22,
        "minimax-video-01": 16,
        "google-veo2": 20,
    }
    creditos_base = creditos_base_map.get(request.modelo or "wavespeed-ai/wan-2.2/t2v-480p", 6)
    creditos_necessarios = creditos_base * request.duracao
    saldo = verificar_saldo(usuario_id, creditos_necessarios)
    if saldo < creditos_necessarios: raise HTTPException(402, f"Créditos insuficientes. Precisa de {creditos_necessarios}.")

    # Sistema inteligente — entende qualquer idioma e otimiza para a API
    tipo_prompt = "thumbnail" if any(w in request.prompt.lower() for w in ["thumbnail","capa","miniatura","tiktok","youtube","reel","viral","click"]) else "imagem"
    
    # Estilo do request enriquece o contexto
    estilo_ctx = ""
    ESTILOS_MAP = {
        "realista": "photorealistic, professional photography, natural lighting, ",
        "cinematico": "cinematic shot, movie scene, dramatic lighting, anamorphic lens, ",
        "anime": "anime style, vibrant illustration, manga art, Studio Ghibli inspired, ",
        "dark": "dark atmosphere, moody, gothic, chiaroscuro lighting, sinister, ",
        "cartoon": "cartoon style, bold outlines, flat colors, comic book, ",
        "3d": "3D render, octane render, volumetric lighting, subsurface scattering, ",
    }
    if hasattr(request, 'estilo') and request.estilo and request.estilo in ESTILOS_MAP:
        estilo_ctx = ESTILOS_MAP[request.estilo]
    
    prompt_base = estilo_ctx + request.prompt if estilo_ctx else request.prompt
    prompt_en = prompt_base + ", highly detailed, 8k uhd, cinematic lighting, award winning"

    # Verificar se é Kling
    if request.modelo and "kling" in request.modelo.lower():
        url = await gerar_video_kling(prompt=prompt_en, duracao=request.duracao, ratio=request.ratio)
        modelo_usado = "Kling AI"
    elif request.modelo and "luma" in request.modelo.lower():
        luma_modelo = "ray-2" if "ray2" in request.modelo.lower() else "dream-machine"
        url = await gerar_video_luma(prompt=prompt_en, duracao=request.duracao, modelo=luma_modelo)
        modelo_usado = "Luma " + luma_modelo
    elif request.modelo and "minimax" in request.modelo.lower():
        url = await gerar_video_minimax(prompt=prompt_en, duracao=request.duracao)
        modelo_usado = "MiniMax Video-01"
    elif request.modelo and "veo" in request.modelo.lower():
        url = await gerar_video_veo(prompt=prompt_en, duracao=request.duracao)
        modelo_usado = "Google Veo 2"
    elif request.modelo and "runway" in request.modelo.lower():
        if not RUNWAY_API_KEY:
            raise HTTPException(500, "RUNWAY_API_KEY não configurada. Adicione sua key do Runway.")
        runway_modelo = "gen3a_turbo" if "turbo" in request.modelo.lower() else "gen3a"
        url = await gerar_video_runway(prompt=prompt_en, duracao=request.duracao, resolucao=request.resolucao, ratio=request.ratio)
        modelo_usado = "Runway Gen-3 " + ("Turbo" if "turbo" in request.modelo.lower() else "Alpha")
    elif FAL_API_KEY:
        try:
            url = await gerar_video_fal(prompt=prompt_en, duracao=request.duracao, modelo="wan")
            modelo_usado = "FAL WAN 2.2"
        except Exception as e_fal:
            erros_vid = [str(e_fal)[:80]]
            if KLING_ACCESS_KEY:
                try:
                    url = await gerar_video_kling(prompt=prompt_en, duracao=min(request.duracao, 5), ratio=request.ratio or "9:16")
                    modelo_usado = "Kling AI (fallback)"
                except Exception as ek:
                    erros_vid.append(str(ek)[:50])
            if not url:
                raise HTTPException(502, f"Vídeo indisponível: {'; '.join(erros_vid)}")
    elif WAVESPEED_API_KEY:
        modelo_ws = request.modelo if request.modelo and "wavespeed" in request.modelo else "wavespeed-ai/wan-2.2/t2v-480p"
        try:
            url = await gerar_video_wavespeed(prompt=prompt_en, duracao=request.duracao, modelo=modelo_ws)
            modelo_usado = "WaveSpeed " + modelo_ws.split("/")[-1]
        except Exception as e_ws:
            if KLING_ACCESS_KEY:
                url = await gerar_video_kling(prompt=prompt_en, duracao=min(request.duracao, 5), ratio=request.ratio or "9:16")
                modelo_usado = "Kling AI (fallback)"
            elif LUMAAI_API_KEY:
                url = await gerar_video_luma(prompt=prompt_en, duracao=request.duracao, modelo="dream-machine")
                modelo_usado = "Luma (fallback)"
            else:
                raise HTTPException(502, f"Vídeo indisponível: {str(e_ws)[:100]}")
    elif RUNWAY_API_KEY:
        url = await gerar_video_runway(prompt=prompt_en, duracao=request.duracao, resolucao=request.resolucao, ratio=request.ratio)
        modelo_usado = "Runway Gen-3"
    else:
        raise HTTPException(500, "Nenhuma API de vídeo configurada.")

    debitar_creditos(usuario_id, creditos_necessarios, "gerar_video")
    return {"ok":True,"video_url":url,"modelo":modelo_usado,"duracao":request.duracao,"prompt_en":prompt_en,"creditos_debitados":creditos_necessarios}


# ══════════════════════════════════════════════════════════════
# ROTA — GERAR VOZ
# ══════════════════════════════════════════════════════════════

@app.post("/clonar-voz")
async def clonar_voz(audio: UploadFile = File(...), nome: str = "Minha Voz", request: Request = None):
    """Clona a voz do usuário via ElevenLabs Instant Voice Cloning."""
    usuario_id = extrair_usuario_id(request) if request else "anon_voice"
    saldo = verificar_saldo(usuario_id, 10)
    if saldo < 10: raise HTTPException(402, "Créditos insuficientes. Precisa de 10.")

    el_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not el_key: raise HTTPException(500, "ELEVENLABS_API_KEY não configurada")

    # Salva o áudio temporariamente
    import shutil, uuid
    audio_path = os.path.join(UPLOAD_DIR, f"voice_{uuid.uuid4().hex}.mp3")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            with open(audio_path, "rb") as af:
                r = await client.post(
                    "https://api.elevenlabs.io/v1/voices/add",
                    headers={"xi-api-key": el_key},
                    data={"name": nome, "description": f"Voz clonada via Vortex AI"},
                    files={"files": (audio.filename or "voice.mp3", af, "audio/mpeg")},
                )
            if not r.is_success:
                raise HTTPException(502, f"ElevenLabs erro {r.status_code}: {r.text[:200]}")
            d = r.json()
            voice_id = d.get("voice_id")
            if not voice_id: raise HTTPException(502, "ElevenLabs não retornou voice_id")

        debitar_creditos(usuario_id, 10, "clonar_voz")
        return {"ok": True, "voice_id": voice_id, "nome": nome}
    finally:
        try: os.remove(audio_path)
        except: pass


class MusicaRequest(BaseModel):
    prompt: str
    duracao: Optional[int] = 15
    estilo: Optional[str] = "cinematic"


@app.post("/gerar-musica")
async def gerar_musica(request: MusicaRequest, req: Request):
    usuario_id = extrair_usuario_id(req)
    creditos = 3
    saldo = verificar_saldo(usuario_id, creditos)
    if saldo < creditos:
        raise HTTPException(402, f"Créditos insuficientes. Precisa de {creditos}.")

    prompt_completo = request.estilo + " music, " + request.prompt + ", high quality audio"
    provedor_musica = request.estilo if request.estilo in ["suno","udio"] else "elevenlabs"
    if provedor_musica == "suno":
        audio_url = await gerar_musica_suno(prompt=prompt_completo, duracao=request.duracao)
    elif provedor_musica == "udio":
        audio_url = await gerar_musica_udio(prompt=prompt_completo)
    else:
        audio_url = await gerar_musica_elevenlabs(prompt=prompt_completo, duracao=request.duracao)
    
    debitar_creditos(usuario_id, creditos, "gerar_musica")
    return {"ok": True, "audio_url": audio_url, "creditos_debitados": creditos}


def _quebrar_texto(texto: str, limite: int = 900) -> list:
    if len(texto) <= limite:
        return [texto]
    chunks, current = [], ""
    for sentence in texto.replace("! ","!|").replace(". ",".|").replace("? ","?|").split("|"):
        if len(current) + len(sentence) <= limite:
            current += sentence + " "
        else:
            if current: chunks.append(current.strip())
            current = sentence + " "
    if current: chunks.append(current.strip())
    return chunks or [texto[:limite]]


@app.post("/gerar-voz")
async def gerar_voz(request: VoiceRequest, req: Request):
    usuario_id = extrair_usuario_id(req)
    chars = len(request.texto)
    creditos_necessarios = max(1, (chars // 1000) * 3)
    saldo = verificar_saldo(usuario_id, creditos_necessarios)
    if saldo < creditos_necessarios: raise HTTPException(402, f"Créditos insuficientes. Precisa de {creditos_necessarios}.")
    audio_url = await gerar_voz_elevenlabs(texto=request.texto, voz_id=request.voz_id, modelo=request.modelo)
    debitar_creditos(usuario_id, creditos_necessarios, "gerar_voz")
    return {"ok":True,"audio_url":audio_url,"modelo":"ElevenLabs","chars":chars,"creditos_debitados":creditos_necessarios}


# ══════════════════════════════════════════════════════════════
# ROTA — TENDÊNCIAS
# ══════════════════════════════════════════════════════════════

# Cache trends — 6 horas
import time as _time
_trends_cache: dict = {}
_trends_cache_ts: dict = {}

@app.get("/tendencias")
async def tendencias(request: Request, nicho: str = "", plataforma: str = "", pais: str = "BR", idioma: str = "pt"):
    """
    Sistema global de tendências — busca tendências reais em tempo real.
    Detecta o país e idioma automaticamente para resultados localizados.
    """
    usuario_id = extrair_usuario_id(request)
    nicho_final = nicho or _perfil.get("nicho", "conteúdo viral")
    plats = _perfil.get("plataformas", [])
    plataforma_final = plataforma or (plats[0] if plats else "TikTok")
    pais_final = pais or "BR"
    
    # Mapa de países para contexto de busca
    PAISES_CONTEXTO = {
        "BR": {"nome": "Brasil", "idioma": "português", "moeda": "BRL"},
        "US": {"nome": "United States", "idioma": "english", "moeda": "USD"},
        "MX": {"nome": "México", "idioma": "español", "moeda": "MXN"},
        "AR": {"nome": "Argentina", "idioma": "español", "moeda": "ARS"},
        "PT": {"nome": "Portugal", "idioma": "português", "moeda": "EUR"},
        "ES": {"nome": "España", "idioma": "español", "moeda": "EUR"},
        "FR": {"nome": "France", "idioma": "français", "moeda": "EUR"},
        "DE": {"nome": "Deutschland", "idioma": "deutsch", "moeda": "EUR"},
    }
    ctx_pais = PAISES_CONTEXTO.get(pais_final, PAISES_CONTEXTO["BR"])
    
    cache_key = f"{nicho_final}_{plataforma_final}_{pais_final}"
    now = _time.time()

    # Cache válido por 3 horas
    if cache_key in _trends_cache and (now - _trends_cache_ts.get(cache_key, 0)) < 10800:
        print(f"[TRENDS] Cache hit: {cache_key}")
        return _trends_cache[cache_key]

    # Buscar tendências reais com Tavily
    trends_reais = []
    # Sistema completo de tendências — Google Trends + Tavily
    try:
        trends_data = await buscar_trends_completo(nicho_final, plataforma_final, pais_final)
        
        # Google Trends como fonte primária
        if trends_data["google_trends"]:
            trends_reais = [
                {
                    "titulo": t,
                    "descricao": f"Viral agora no Google Brasil — use isso como hook!",
                    "como_usar": f"Crie conteúdo de {nicho_final} conectando com '{t}'",
                    "potencial": "alto",
                    "hashtags": [f"#{t.replace(' ', '')}", f"#{nicho_final}"],
                    "fonte": "Google Trends"
                }
                for t in trends_data["google_trends"][:5]
            ]
            print(f"[TRENDS] ✅ Google Trends: {len(trends_reais)} trends")
    except Exception as e_gt:
        print(f"[TRENDS] Google Trends falhou: {e_gt}")

    if TAVILY_API_KEY and not trends_reais:
        try:
            query = f"tendências virais {nicho_final} {plataforma_final} {ctx_pais['nome']} 2026"
            resultado_tavily = await buscar_tavily(query, max_results=5)
            
            if resultado_tavily:
                # Processar com IA para extrair trends estruturadas
                prompt_trends = f"""Você é um analista de tendências virais do Brasil.

Com base nos dados abaixo, extraia 5 tendências REAIS e ESPECÍFICAS de {nicho_final} no {ctx_pais['nome']}.

DADOS:
{resultado_tavily[:1500]}

REGRAS OBRIGATÓRIAS:
- Use APENAS informações dos dados acima — não invente
- Cada tendência deve ter um nome ESPECÍFICO (ex: "BBB 2026", "Free Fire novo personagem") não genérico ("Conteúdo viral")
- Se não houver dados suficientes, reduza para 3 tendências reais em vez de inventar 5

Responda APENAS com JSON:
[
  {{
    "titulo": "Nome específico e real da tendência",
    "descricao": "Por que está viralizando agora — baseado nos dados",
    "como_usar": "Como creator de {nicho_final} pode usar",
    "potencial": "alto/médio/baixo",
    "hashtags": ["#tag1", "#tag2", "#tag3"]
  }}
]"""

                msgs = [{"role": "user", "content": prompt_trends}]
                resultado_ia, _ = await gerar_texto(msgs, 
                    system="Você extrai tendências virais em JSON estruturado. Responda APENAS com JSON válido.",
                    max_tokens=1000, provedor_preferido="groq")
                
                import json as _json
                try:
                    clean = resultado_ia.strip().replace("```json","").replace("```","").strip()
                    trends_reais = _json.loads(clean)
                except:
                    trends_reais = []
        except Exception as e:
            print(f"[TRENDS] Tavily falhou: {e}")

    # Fallback para trends estáticas se tudo falhar
    if not trends_reais:
        nicho_data = TENDENCIAS_2026.get(nicho_final.lower(), TENDENCIAS_2026.get("lifestyle", {}))
        trends_estaticas = nicho_data.get(plataforma_final, ["Conteúdo autêntico", "POV viral", "Storytelling"])
        trends_reais = [{"titulo": t, "descricao": "Tendência em alta", "potencial": "alto", "hashtags": []} for t in trends_estaticas[:5]]

    resp = {
        "ok": True,
        "nicho": nicho_final,
        "plataforma": plataforma_final,
        "pais": ctx_pais["nome"],
        "idioma": ctx_pais["idioma"],
        "tendencias": trends_reais,
        "fonte": "real_time" if TAVILY_API_KEY else "base_dados",
        "atualizado_em": now
    }
    
    _trends_cache[cache_key] = resp
    _trends_cache_ts[cache_key] = now
    return resp


# ══════════════════════════════════════════════════════════════
# ROTA — CRÉDITOS
# ══════════════════════════════════════════════════════════════

class AvatarRequest(BaseModel):
    imagem_url: str
    audio_url: str


@app.post("/gerar-avatar")
async def gerar_avatar(request: AvatarRequest, req: Request):
    usuario_id = extrair_usuario_id(req)
    creditos = 20
    saldo = verificar_saldo(usuario_id, creditos)
    if saldo < creditos:
        raise HTTPException(402, f"Créditos insuficientes. Precisa de {creditos}.")
    url = await gerar_avatar_hedra(imagem_url=request.imagem_url, audio_url=request.audio_url)
    debitar_creditos(usuario_id, creditos, "gerar_avatar")
    return {"ok": True, "video_url": url, "creditos_debitados": creditos}


@app.get("/creditos/saldo")
def creditos_saldo(request: Request):
    uid = extrair_usuario_id(request)
    return {"ok":True,"saldo":get_saldo(uid),"usuario_id":uid}

@app.get("/creditos/historico")
def creditos_hist(request: Request):
    uid = extrair_usuario_id(request)
    return {"ok":True,"historico":historico_creditos(uid)}


# ══════════════════════════════════════════════════════════════
# ROTA — PAGAMENTO (Mercado Pago)
# ══════════════════════════════════════════════════════════════

PLANOS_CONFIG = {
    "starter_mensal":  {"creditos": 250,  "preco": 9.00,  "nome": "Starter Mensal"},
    "creator_mensal":  {"creditos": 900,  "preco": 27.00, "nome": "Creator Mensal"},
    "pro_mensal":      {"creditos": 2500, "preco": 57.00, "nome": "Pro Mensal"},
    "elite_mensal":    {"creditos": 8000, "preco": 97.00, "nome": "Elite Mensal"},
    "starter_anual":   {"creditos": 300,  "preco": 84.00, "nome": "Starter Anual"},
    "creator_anual":   {"creditos": 1080, "preco": 264.00,"nome": "Creator Anual"},
    "pro_anual":       {"creditos": 3000, "preco": 552.00,"nome": "Pro Anual"},
    "elite_anual":     {"creditos": 9600, "preco": 936.00,"nome": "Elite Anual"},
    "pack_mini":       {"creditos": 150,  "preco": 9.00,  "nome": "Pack Mini"},
    "pack_creator":    {"creditos": 600,  "preco": 29.00, "nome": "Pack Creator"},
    "pack_pro":        {"creditos": 1800, "preco": 69.00, "nome": "Pack Pro"},
    "pack_max":        {"creditos": 5000, "preco": 129.00,"nome": "Pack Max"},
}

class PagamentoRequest(BaseModel):
    plano_id: str
    usuario_id: Optional[str] = "default"


@app.post("/criar-pagamento")
async def criar_pagamento(request: PagamentoRequest):
    mp_token = os.getenv("MP_ACCESS_TOKEN", "")
    if not mp_token:
        raise HTTPException(500, "MP_ACCESS_TOKEN não configurado")

    plano = PLANOS_CONFIG.get(request.plano_id)
    if not plano:
        raise HTTPException(400, f"Plano inválido: {request.plano_id}")

    vortex_url = os.getenv("VORTEX_URL", "http://127.0.0.1:8082")

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(
            "https://api.mercadopago.com/checkout/preferences",
            headers={"Authorization": f"Bearer {mp_token}", "Content-Type": "application/json"},
            json={
                "items": [{
                    "title": f"Vortex AI — {plano['nome']}",
                    "quantity": 1,
                    "unit_price": plano["preco"],
                    "currency_id": "BRL",
                }],
                "back_urls": {
                    "success": f"{vortex_url}/pagamento/sucesso?plano={request.plano_id}&usuario={request.usuario_id}",
                    "failure": f"{vortex_url}/pagamento/falha",
                    "pending": f"{vortex_url}/pagamento/pendente",
                },
                "auto_return": "approved",
                "external_reference": f"{request.usuario_id}_{request.plano_id}",
                "metadata": {"usuario_id": request.usuario_id, "plano_id": request.plano_id},
            },
        )
        if not r.is_success:
            raise HTTPException(502, f"Mercado Pago erro: {r.text[:200]}")
        d = r.json()
        return {
            "ok": True,
            "checkout_url": d.get("init_point"),
            "preference_id": d.get("id"),
        }


@app.get("/pagamento/sucesso")
async def pagamento_sucesso(request: Request, plano: str, usuario: str = "", payment_id: str = ""):
    from fastapi.responses import RedirectResponse
    plano_cfg = PLANOS_CONFIG.get(plano)
    if plano_cfg:
        # Adiciona créditos ao usuário
        creditos_atuais = get_creditos_db(usuario)
        novos_creditos = creditos_atuais + plano_cfg["creditos"]
        set_creditos_db(usuario, novos_creditos)
        salvar_usuario_db(usuario, {"plano": plano})
        salvar_geracao_db(usuario, "pagamento", {
            "plano": plano, "creditos": plano_cfg["creditos"],
            "preco": plano_cfg["preco"], "payment_id": payment_id
        })
        print(f"[Pagamento] ✅ {usuario} comprou {plano} — +{plano_cfg['creditos']} créditos")

    frontend = os.getenv("FRONTEND_URL", "http://localhost:5173")
    return RedirectResponse(f"{frontend}?pagamento=ok&plano={plano}")


@app.post("/webhook/mercadopago")
async def webhook_mp(request: Request):
    """Webhook do Mercado Pago com validação de assinatura"""
    # Validar assinatura MP — evita webhooks falsos
    mp_secret = os.getenv("MP_WEBHOOK_SECRET", "")
    raw_body = await request.body()
    if mp_secret:
        try:
            sig = request.headers.get("x-signature", "")
            req_id = request.headers.get("x-request-id", "")
            parts = dict(p.split("=", 1) for p in sig.split(",") if "=" in p)
            ts = parts.get("ts", "")
            v1 = parts.get("v1", "")
            if ts and v1:
                manifest = f"id:{req_id};request-id:{req_id};ts:{ts};"
                expected = hmac.new(
                    mp_secret.encode(), manifest.encode(), hashlib.sha256
                ).hexdigest()
                if not hmac.compare_digest(expected, v1):
                    print("[MP Webhook] ⚠️ Assinatura inválida!")
                    raise HTTPException(401, "Assinatura inválida")
        except HTTPException:
            raise
        except Exception as e:
            print(f"[MP Webhook] Erro validação: {e}")

    try:
        data = json.loads(raw_body)
    except:
        data = {}
    print(f"[MP Webhook] type={data.get('type')} id={data.get('data',{}).get('id')}")
    tipo = data.get("type")
    if tipo == "payment":
        payment_id = data.get("data", {}).get("id")
        if payment_id:
            mp_token = os.getenv("MP_ACCESS_TOKEN", "")
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"https://api.mercadopago.com/v1/payments/{payment_id}",
                    headers={"Authorization": f"Bearer {mp_token}"},
                )
                if r.is_success:
                    p = r.json()
                    if p.get("status") == "approved":
                        ref = p.get("external_reference", "")
                        parts = ref.split("_", 1)
                        if len(parts) == 2:
                            usuario_id, plano_id = parts
                            plano_cfg = PLANOS_CONFIG.get(plano_id)
                            if plano_cfg:
                                # IDEMPOTÊNCIA — evita processar o mesmo pagamento 2x
                                user_data = get_usuario_db(usuario_id)
                                pagamentos_processados = user_data.get("pagamentos_processados", [])

                                if payment_id in pagamentos_processados:
                                    print(f"[MP Webhook] ⚠️ Pagamento {payment_id} já processado — ignorando")
                                else:
                                    # Adicionar créditos
                                    creditos_atuais = get_creditos_db(usuario_id)
                                    set_creditos_db(usuario_id, creditos_atuais + plano_cfg["creditos"])

                                    # Registrar pagamento como processado
                                    pagamentos_processados.append(payment_id)
                                    # Manter só os últimos 50 pagamentos
                                    user_data["pagamentos_processados"] = pagamentos_processados[-50:]
                                    salvar_usuario_db(usuario_id, user_data)

                                    print(f"[MP Webhook] ✅ Créditos adicionados: {usuario_id} +{plano_cfg['creditos']} (payment={payment_id})")
    return {"ok": True}

# ══════════════════════════════════════════════════════════════
# ROTA — UPLOAD DE VÍDEO
# ══════════════════════════════════════════════════════════════

@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...), request: Request = None):
    # Salva o arquivo localmente
    ext = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(video.file, f)
    
    # Tamanho e duração estimada
    size_mb = os.path.getsize(filepath) / (1024*1024)
    duracao_estimada = max(5, min(120, size_mb * 8))  # estimativa: ~8s por MB
    
    # URL pública acessível pelo Shotstack
    # Em produção: use Cloudflare R2, S3 ou outro CDN
    # Por agora: URL local via staticfiles (funciona se Shotstack acessar localhost)
    # Para produção real: fazer ngrok ou deploy em servidor público
    host = os.getenv("VORTEX_URL", "http://localhost:8082")
    url_publica = f"{host}/uploads/{filename}"
    
    print(f"[UPLOAD] {filename} salvo ({size_mb:.1f}MB) → {url_publica}")
    return {
        "ok": True,
        "url": url_publica,
        "filename": filename,
        "duracao": duracao_estimada,
        "size_mb": round(size_mb, 2),
    }


# ══════════════════════════════════════════════════════════════
# ROTA — EDIÇÃO DE VÍDEO (Shotstack)
# ══════════════════════════════════════════════════════════════

@app.post("/editar-video")
async def editar_video(request: EditarVideoRequest, req: Request):
    usuario_id = extrair_usuario_id(req)
    saldo = verificar_saldo(usuario_id, 10)
    if saldo < 10: raise HTTPException(402, "Créditos insuficientes. Precisa de 10.")
    
    ss_sandbox = os.getenv("SHOTSTACK_SANDBOX_KEY", "")
    ss_prod    = os.getenv("SHOTSTACK_PROD_KEY", "")
    ss_key     = ss_sandbox or ss_prod
    ss_env     = "stage" if ss_sandbox else "v1"
    
    if not ss_key:
        raise HTTPException(500, "Shotstack não configurado. Adicione SHOTSTACK_SANDBOX_KEY no .bat")

    import json as _json

    # Monta payload correto para o Shotstack
    # O frontend manda o objeto interno do timeline
    raw = request.timeline
    
    # Se já tem "timeline" e "output" como keys, usa direto
    if "timeline" in raw and "output" in raw:
        payload = raw
    # Se tem "tracks" diretamente, envolve no formato correto
    elif "tracks" in raw:
        payload = {
            "timeline": {"tracks": raw["tracks"]},
            "output": {"format": "mp4", "resolution": "sd"}
        }
    else:
        payload = {
            "timeline": raw,
            "output": {"format": "mp4", "resolution": "sd"}
        }

    print(f"[Shotstack] env={ss_env} key={ss_key[:8]}...")
    print(f"[Shotstack] payload: {_json.dumps(payload)[:500]}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(
            f"https://api.shotstack.io/{ss_env}/render",
            json=payload,
            headers={"Content-Type":"application/json","x-api-key":ss_key},
        )
        if not r.is_success:
            raise HTTPException(502, f"Shotstack erro {r.status_code}: {r.text[:300]}")
        data = r.json()
        render_id = data.get("response",{}).get("id")
        if not render_id:
            raise HTTPException(502, f"Shotstack sem render_id: {data}")

    # Polling
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        for i in range(60):
            await asyncio.sleep(4)
            poll = await client.get(
                f"https://api.shotstack.io/{ss_env}/render/{render_id}",
                headers={"x-api-key":ss_key},
            )
            d = poll.json().get("response",{})
            status = d.get("status")
            print(f"[Shotstack] status={status} ({i*4}s)")
            if status == "done":
                url = d.get("url")
                if url:
                    debitar_creditos(usuario_id, 10, "editar_video")
                    return {"ok":True,"video_url":url,"render_id":render_id}
            if status == "failed":
                raise HTTPException(502, f"Shotstack falhou: {d.get('error','desconhecido')}")

    raise HTTPException(504, "Timeout na renderização Shotstack (4 min)")


@app.post("/transcrever")
async def transcrever(request: Request):
    # Endpoint placeholder — transcrição futura via Whisper
    return {"ok": True, "words": [], "texto": ""}


@app.post("/pagamento/criar")
async def criar_pagamento(request: PagamentoRequest):
    if not MP_ACCESS_TOKEN: raise HTTPException(500, "MP_ACCESS_TOKEN não configurado.")
    try:
        import mercadopago
        mp = mercadopago.SDK(MP_ACCESS_TOKEN)
    except ImportError:
        raise HTTPException(500, "Instale: pip install mercadopago")
    preference_data = {
        "items":[{"title":request.descricao,"quantity":1,"unit_price":float(request.preco),"currency_id":"BRL"}],
        "back_urls":{"success":f"{VORTEX_URL}/creditos/sucesso","failure":f"{VORTEX_URL}/creditos","pending":f"{VORTEX_URL}/creditos"},
        "auto_return":"approved",
        "metadata":{"pacote_id":request.pacote_id,"creditos":request.creditos},
    }
    result = mp.preference().create(preference_data)
    if result["status"] != 201: raise HTTPException(502, f"Mercado Pago erro: {result}")
    preference = result["response"]
    checkout_url = preference.get("sandbox_init_point") or preference.get("init_point")
    return {"checkout_url":checkout_url,"preference_id":preference["id"]}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

# ── Tavily Web Search — busca fatos reais para roteiros ──────────────────────
async def buscar_google_trends(nicho: str = "", pais: str = "BR") -> list:
    """
    Busca tendências reais — múltiplas fontes com fallback.
    1. Google Trends RSS (grátis)
    2. Google Trends API alternativa
    3. Tavily com query específica
    """
    import re as _re

    trends = []

    # FONTE 1: Google Trends RSS oficial
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            r = await client.get(
                f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={pais}",
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/rss+xml, application/xml, text/xml",
                    "Accept-Language": "pt-BR,pt;q=0.9",
                },
            )
            if r.is_success and "<title>" in r.text:
                items = _re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', r.text)
                items = [i.strip() for i in items if i and "Google" not in i and len(i) > 2][:10]
                if items:
                    print(f"[TRENDS] ✅ Google RSS: {len(items)} trends")
                    return items
    except Exception as e:
        print(f"[TRENDS] RSS falhou: {e}")

    # FONTE 2: Google Trends via endpoint alternativo
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            r = await client.get(
                "https://trends.google.com/trends/api/dailytrends",
                params={"hl": "pt-BR", "tz": "-180", "geo": pais, "ns": "15"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.is_success:
                # Remove o prefix de segurança do Google
                text = r.text
                if text.startswith(")]}',"):
                    text = text[5:]
                import json as _json2
                data = _json2.loads(text)
                items = []
                for day in data.get("default", {}).get("trendingSearchesDays", [])[:1]:
                    for search in day.get("trendingSearches", [])[:10]:
                        title = search.get("title", {}).get("query", "")
                        if title:
                            items.append(title)
                if items:
                    print(f"[TRENDS] ✅ Google API: {len(items)} trends")
                    return items
    except Exception as e:
        print(f"[TRENDS] API falhou: {e}")

    # FONTE 3: Tavily com query específica para trends reais
    if TAVILY_API_KEY:
        try:
            from datetime import datetime
            hoje = datetime.now().strftime("%d/%m/%Y")
            query = f"o que está viral no Brasil hoje {hoje} TikTok trending"
            resultado = await buscar_tavily(query, max_results=5)
            if resultado:
                # Extrair nomes específicos do resultado
                nomes = _re.findall(r'"([^"]{5,50})"', resultado)
                nomes = [n for n in nomes if not any(w in n.lower() for w in ["http","www","com","br","the","and"])][:8]
                if nomes:
                    print(f"[TRENDS] ✅ Tavily: {len(nomes)} trends")
                    return nomes
        except Exception as e:
            print(f"[TRENDS] Tavily falhou: {e}")

    print("[TRENDS] ⚠️ Todas as fontes falharam")
    return []


async def buscar_trends_completo(nicho: str, plataforma: str, pais: str = "BR") -> dict:
    """
    Sistema completo de tendências — combina Google Trends + Tavily + base interna.
    Retorna o contexto mais rico possível para o roteiro.
    """
    resultado = {
        "google_trends": [],
        "tavily_trends": [],
        "base_interna": [],
        "contexto_completo": "",
    }
    
    # 1. Google Trends — o que está viral agora no Brasil
    google = await buscar_google_trends(nicho, pais)
    resultado["google_trends"] = google[:5]
    
    # 2. Tavily — análise profunda com contexto
    if TAVILY_API_KEY:
        try:
            query = f"tendências virais {nicho} {plataforma} Brasil 2026 agora"
            tavily = await buscar_tavily(query, max_results=5)
            resultado["tavily_trends"] = [tavily] if tavily else []
        except Exception as e:
            print(f"[TRENDS] Tavily falhou: {e}")
    
    # 3. Montar contexto completo para a IA usar
    partes = []
    
    if google:
        header = "VIRAL NO BRASIL AGORA (Google Trends):"
        items = "\n".join(f"• {t}" for t in google[:5])
        partes.append(f"{header}\n{items}")
    
    if resultado["tavily_trends"]:
        analise = resultado["tavily_trends"][0][:500]
        partes.append(f"ANÁLISE DE TENDÊNCIAS {plataforma.upper()}:\n{analise}")
    
    resultado["contexto_completo"] = "\n\n".join(partes)
    
    print(f"[TRENDS-COMPLETO] ✅ {len(google)} Google + Tavily combinados")
    return resultado


async def buscar_tavily(query: str, max_results: int = 3) -> str:
    """Busca informações reais na web via Tavily AI."""
    key = os.getenv("TAVILY_API_KEY", TAVILY_API_KEY)
    if not key:
        return ""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json"},
                json={
                    "api_key": key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False,
                    "language": "pt",
                },
            )
            if not r.is_success:
                print(f"[Tavily] erro {r.status_code}: {r.text[:100]}")
                return ""
            d = r.json()
            partes = []
            # Resposta direta
            if d.get("answer"):
                partes.append(d["answer"])
            # Resultados individuais
            for res in d.get("results", [])[:max_results]:
                if res.get("content"):
                    partes.append(f"[{res.get('title','')}]: {res['content'][:400]}")
            resultado = "\n\n".join(partes)
            print(f"[Tavily] ✅ {len(resultado)} chars sobre '{query}'")
            return resultado
    except Exception as e:
        print(f"[Tavily] falhou: {e}")
        return ""


# ── WaveSpeed direto (alternativa ao Leonardo/Runway) ────────────────────────
async def gerar_imagem_gemini(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """Gera imagem via Gemini 2.0 Flash — gratuito."""
    import base64
    key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise HTTPException(500, "GEMINI_API_KEY não configurada")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-preview-image-generation:generateContent?key={key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]}
            }
        )
        if not r.is_success:
            raise HTTPException(502, f"Gemini erro {r.status_code}: {r.text[:200]}")
        d = r.json()
        # Extrair imagem da resposta
        for part in d.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if part.get("inlineData"):
                img_b64 = part["inlineData"]["data"]
                mime = part["inlineData"].get("mimeType", "image/jpeg")
                return f"data:{mime};base64,{img_b64}"
        raise HTTPException(502, f"Gemini sem imagem na resposta: {str(d)[:200]}")


async def gerar_imagem_hf(prompt: str) -> str:
    """Gera imagem via Hugging Face — FLUX.1-schnell gratuito."""
    import base64
    key = os.getenv("HF_API_KEY", "")
    if not key:
        raise HTTPException(500, "HF_API_KEY não configurada")
    
    MODELOS_IMG = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "ByteDance/SDXL-Lightning",
    ]
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        for modelo in MODELOS_IMG:
            try:
                r = await client.post(
                    f"https://api-inference.huggingface.co/models/{modelo}",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"inputs": prompt},
                )
                if r.is_success and r.headers.get("content-type","").startswith("image"):
                    img_b64 = base64.b64encode(r.content).decode()
                    print(f"[HF IMG] ✅ {modelo}")
                    return f"data:image/jpeg;base64,{img_b64}"
                print(f"[HF IMG] {modelo}: {r.status_code} {r.text[:80]}")
            except Exception as e:
                print(f"[HF IMG] {modelo} erro: {e}")
    raise HTTPException(502, "HuggingFace imagem indisponível")


async def gerar_video_hf(prompt: str) -> str:
    """Gera vídeo via Hugging Face — LTX-Video gratuito."""
    import base64
    key = os.getenv("HF_API_KEY", "")
    if not key:
        raise HTTPException(500, "HF_API_KEY não configurada")
    
    MODELOS_VID = [
        "Lightricks/LTX-Video",
        "damo-vilab/text-to-video-ms-1.7b",
        "cerspense/zeroscope_v2_576w",
    ]
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        for modelo in MODELOS_VID:
            try:
                r = await client.post(
                    f"https://api-inference.huggingface.co/models/{modelo}",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"inputs": prompt},
                )
                if r.is_success and r.headers.get("content-type","").startswith("video"):
                    vid_b64 = base64.b64encode(r.content).decode()
                    print(f"[HF VID] ✅ {modelo}")
                    return f"data:video/mp4;base64,{vid_b64}"
                print(f"[HF VID] {modelo}: {r.status_code} {r.text[:80]}")
            except Exception as e:
                print(f"[HF VID] {modelo} erro: {e}")
    raise HTTPException(502, "HuggingFace vídeo indisponível")


async def gerar_imagem_fal(prompt: str, modelo: str = "fal-ai/flux/dev", width: int = 1024, height: int = 1024) -> str:
    """Gera imagem via FAL.ai — Flux Dev, Flux Pro, SDXL e mais."""
    key = FAL_API_KEY or os.getenv("FAL_API_KEY", "")
    if not key:
        raise HTTPException(500, "FAL_API_KEY não configurada")

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        # Submete o job
        r = await client.post(
            f"https://queue.fal.run/{modelo}",
            headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
            json={"prompt": prompt, "image_size": {"width": width, "height": height}, "num_images": 1, "output_format": "jpeg"},
        )
        if not r.is_success:
            raise HTTPException(502, f"FAL imagem erro {r.status_code}: {r.text[:200]}")
        d = r.json()
        request_id = d.get("request_id")
        if not request_id:
            raise HTTPException(502, f"FAL sem request_id: {d}")

    # Polling
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        for i in range(40):
            await asyncio.sleep(3)
            poll = await client.get(
                f"https://queue.fal.run/{modelo}/requests/{request_id}/status",
                headers={"Authorization": f"Key {key}"},
            )
            pd = poll.json()
            status = pd.get("status")
            print(f"[FAL] img status={status} ({i*3}s)")
            if status == "COMPLETED":
                result = await client.get(
                    f"https://queue.fal.run/{modelo}/requests/{request_id}",
                    headers={"Authorization": f"Key {key}"},
                )
                rd = result.json()
                images = rd.get("images", [])
                if images:
                    return images[0].get("url", "")
            if status == "FAILED":
                raise HTTPException(502, f"FAL imagem falhou: {pd}")
    raise HTTPException(504, "FAL imagem timeout")


async def gerar_video_fal(prompt: str, duracao: int = 5, modelo: str = "fal-ai/wan-t2v") -> str:
    """Gera vídeo via FAL.ai — WAN 2.2, Kling, e mais."""
    key = FAL_API_KEY or os.getenv("FAL_API_KEY", "")
    if not key:
        raise HTTPException(500, "FAL_API_KEY não configurada")

    MODELOS_FAL = {
        "wan": "fal-ai/wan-t2v",
        "kling": "fal-ai/kling-video/v1.6/standard/text-to-video",
        "minimax": "fal-ai/minimax-video/image-to-video",
    }
    modelo_final = MODELOS_FAL.get(modelo, modelo)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        r = await client.post(
            f"https://queue.fal.run/{modelo_final}",
            headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
            json={"prompt": prompt, "duration": str(duracao)},
        )
        if not r.is_success:
            raise HTTPException(502, f"FAL vídeo erro {r.status_code}: {r.text[:200]}")
        d = r.json()
        request_id = d.get("request_id")
        if not request_id:
            raise HTTPException(502, f"FAL sem request_id: {d}")

    # Polling
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        for i in range(60):
            await asyncio.sleep(5)
            poll = await client.get(
                f"https://queue.fal.run/{modelo_final}/requests/{request_id}/status",
                headers={"Authorization": f"Key {key}"},
            )
            pd = poll.json()
            status = pd.get("status")
            print(f"[FAL] video status={status} ({i*5}s)")
            if status == "COMPLETED":
                result = await client.get(
                    f"https://queue.fal.run/{modelo_final}/requests/{request_id}",
                    headers={"Authorization": f"Key {key}"},
                )
                rd = result.json()
                video = rd.get("video", {})
                return video.get("url", "")
            if status == "FAILED":
                raise HTTPException(502, f"FAL vídeo falhou")
    raise HTTPException(504, "FAL vídeo timeout")


async def gerar_imagem_wavespeed(prompt: str, endpoint: str = "wavespeed-ai/flux-dev") -> str:
    if not WAVESPEED_API_KEY:
        raise HTTPException(500, "WAVESPEED_API_KEY não configurada")
    ep = endpoint or "wavespeed-ai/flux-dev"
    
    # Tenta sync mode primeiro (resultado imediato)
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(
            f"https://api.wavespeed.ai/api/v3/{ep}",
            headers={"Authorization": f"Bearer {WAVESPEED_API_KEY}", "Content-Type": "application/json"},
            json={
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "num_images": 1,
                "seed": -1,
                "enable_sync_mode": True,
                "output_format": "jpeg"
            },
        )
        if not r.is_success:
            raise HTTPException(502, f"WaveSpeed imagem erro {r.status_code}: {r.text[:300]}")
        d = r.json()
        print(f"[WaveSpeed] resposta: {str(d)[:200]}")
        
        # Modo sync — resultado já veio
        data = d.get("data", {})
        outputs = data.get("outputs", [])
        if outputs:
            return outputs[0]
        
        # Modo async — polling necessário
        request_id = data.get("id")
        if not request_id:
            raise HTTPException(502, f"WaveSpeed sem outputs e sem request_id: {d}")
        print(f"[WaveSpeed] Imagem job async: {request_id}")

    # Polling
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        for i in range(30):
            await asyncio.sleep(3)
            poll = await client.get(
                f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result",
                headers={"Authorization": f"Bearer {WAVESPEED_API_KEY}"},
            )
            pd = poll.json()
            status = pd.get("data", {}).get("status")
            print(f"[WaveSpeed] img status={status} ({i*3}s)")
            if status == "completed":
                outputs = pd.get("data", {}).get("outputs", [])
                if outputs: return outputs[0]
            if status == "failed":
                raise HTTPException(502, f"WaveSpeed imagem falhou")
    raise HTTPException(504, "WaveSpeed imagem timeout")

async def gerar_video_wavespeed(prompt: str, duracao: int = 5, modelo: str = "wavespeed-ai/wan-2.2/t2v-480p") -> str:
    if not WAVESPEED_API_KEY:
        raise HTTPException(500, "WAVESPEED_API_KEY não configurada")
    # Tamanho baseado no modelo
    size = "720*1280" if "720p" in modelo else "480*832"
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(
            f"https://api.wavespeed.ai/api/v3/{modelo}",
            headers={"Authorization": f"Bearer {WAVESPEED_API_KEY}", "Content-Type": "application/json"},
            json={"prompt": prompt, "duration": duracao, "size": size},
        )
        if not r.is_success: raise HTTPException(502, f"WaveSpeed vídeo erro {r.status_code}: {r.text[:200]}")
        d = r.json()
        request_id = d.get("data", {}).get("id")
        if not request_id: raise HTTPException(502, "WaveSpeed não retornou request_id")

    # Polling
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        for i in range(40):
            await asyncio.sleep(3)
            poll = await client.get(
                f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result",
                headers={"Authorization": f"Bearer {WAVESPEED_API_KEY}"},
            )
            pd = poll.json()
            status = pd.get("data", {}).get("status")
            if status == "completed":
                url = pd.get("data", {}).get("outputs", [None])[0]
                if url: return url
            if status == "failed":
                raise HTTPException(502, "WaveSpeed vídeo falhou")
    raise HTTPException(504, "WaveSpeed timeout")



# ── Endpoint /me — retorna tudo do usuário de uma vez ─────────────────────────


# ══════════════════════════════════════════════════════════════════
# 🎬 VIDEO FACELESS AUTOMÁTICO — O maior diferencial do Vortex
# Fluxo: Tema → Roteiro → Prompts de imagem → Imagens → Narração → Vídeo final
# ══════════════════════════════════════════════════════════════════
class VideoFacelessRequest(BaseModel):
    tema: str
    nicho: Optional[str] = "viral"
    duracao: Optional[int] = 60
    estilo: Optional[str] = "cinematico"
    voz_id: Optional[str] = "onwK4e9ZLuTAKqWW03F9"
    plataforma: Optional[str] = "TikTok"

@app.post("/video-faceless")
async def gerar_video_faceless(request: VideoFacelessRequest):
    """
    Gera um vídeo faceless completo em 1 clique:
    1. Gera roteiro viral do tema
    2. Extrai cenas e prompts de imagem
    3. Gera imagens para cada cena
    4. Gera narração com ElevenLabs
    5. Retorna tudo pronto para montar
    """
    resultado = {
        "ok": True,
        "tema": request.tema,
        "etapas": {}
    }

    # ETAPA 1 — Gerar roteiro
    try:
        system = VORTEX_CRIADOR
        prompt_roteiro = f"""Crie um roteiro faceless viral de {request.duracao} segundos para {request.plataforma}.

TEMA: {request.tema}
NICHO: {request.nicho}
ESTILO: {request.estilo}

FORMATO ESPECIAL PARA VÍDEO FACELESS:
Divida em exatamente 5 CENAS numeradas.
Para cada cena forneça:
- NARRAÇÃO: texto exato que será narrado
- VISUAL: descrição detalhada da imagem (em inglês, para gerar com IA)
- DURAÇÃO: segundos desta cena

Formato obrigatório:
CENA 1 ({request.duracao//5}s)
NARRAÇÃO: [texto]
VISUAL: [descrição em inglês para gerar imagem]

CENA 2 ...

Ao final, inclua:
TÍTULO: [título viral]
LEGENDA: [legenda para TikTok]
HASHTAGS: [hashtags]"""

        msgs = [{"role": "user", "content": prompt_roteiro}]
        roteiro, _ = await gerar_texto_roteiro(msgs, system=system, max_tokens=2000)
        resultado["etapas"]["roteiro"] = roteiro
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar roteiro: {str(e)[:100]}")

    # ETAPA 2 — Extrair cenas e gerar imagens
    import re as _re
    cenas = []
    visuais = _re.findall(r'VISUAL:\s*(.+?)(?=CENA|\Z)', roteiro, _re.DOTALL)
    narracoes = _re.findall(r'NARRAÇÃO:\s*(.+?)(?=VISUAL|CENA|\Z)', roteiro, _re.DOTALL)

    imagens_geradas = []
    for i, visual in enumerate(visuais[:5]):
        try:
            visual_limpo = visual.strip()[:300]
            # Enriquecer o prompt visual
            prompt_img = await interpretar_prompt_inteligente(visual_limpo, tipo="video")
            img_url = await gerar_imagem_wavespeed(prompt_img, modelo="wavespeed-ai/flux-dev")
            imagens_geradas.append({
                "cena": i + 1,
                "prompt": prompt_img,
                "url": img_url,
                "narracao": narracoes[i].strip() if i < len(narracoes) else ""
            })
        except Exception as e:
            imagens_geradas.append({
                "cena": i + 1,
                "erro": str(e)[:50],
                "narracao": narracoes[i].strip() if i < len(narracoes) else ""
            })

    resultado["etapas"]["imagens"] = imagens_geradas

    # ETAPA 3 — Gerar narração completa
    try:
        texto_narracao = " ".join([n.strip() for n in narracoes if n.strip()])
        if texto_narracao:
            audio_url = await gerar_voz_elevenlabs(texto_narracao[:1000], voz_id=request.voz_id)
            resultado["etapas"]["audio"] = audio_url
    except Exception as e:
        resultado["etapas"]["audio_erro"] = str(e)[:100]

    # ETAPA 4 — Extrair metadados
    titulo_match = _re.search(r'TÍTULO:\s*(.+)', roteiro)
    legenda_match = _re.search(r'LEGENDA:\s*(.+)', roteiro)
    hashtags_match = _re.search(r'HASHTAGS:\s*(.+)', roteiro)

    resultado["metadados"] = {
        "titulo": titulo_match.group(1).strip() if titulo_match else request.tema,
        "legenda": legenda_match.group(1).strip() if legenda_match else "",
        "hashtags": hashtags_match.group(1).strip() if hashtags_match else "",
        "plataforma": request.plataforma,
        "duracao": request.duracao
    }

    return resultado


@app.post("/gerar-imagem-free")
@limiter.limit("20/minute")  # max 20 imagens por minuto por IP
async def gerar_imagem_free(request: ImageRequest, req: Request):
    """Endpoint gratuito — Gemini Imagen 3 → HuggingFace → Pollinations."""
    usuario_id = extrair_usuario_id(req, request)
    import urllib.parse, base64
    
    prompt_en = request.prompt + ", highly detailed, cinematic, 8k uhd, professional photography"
    modelo_req = request.modelo or ""

    modelo_req = request.modelo or "hf_flux"
    hf_key = os.getenv("HF_API_KEY", "")

    # Cascata de imagem grátis — tenta em ordem até funcionar
    print(f"[IMG-FREE] modelo={modelo_req} hf_key={bool(hf_key)}")
    erros_img = []

    # 1. HuggingFace FLUX (melhor qualidade grátis)
    if modelo_req in ("hf_flux", "") and hf_key:
        try:
            url = await gerar_imagem_hf(prompt_en)
            return {"ok": True, "imagem": url, "modelo": "🤗 FLUX Schnell", "prompt_en": prompt_en}
        except Exception as e:
            erros_img.append(f"HF: {e}")
            print(f"[HF] falhou: {e}")

    # 2. Gemini Image (grátis com nossa key)
    try:
        url = await gerar_imagem_gemini(prompt_en)
        return {"ok": True, "imagem": url, "modelo": "✨ Gemini Image", "prompt_en": prompt_en}
    except Exception as e:
        erros_img.append(f"Gemini: {e}")
        print(f"[Gemini] falhou: {e}")

    # 3. Pollinations (sempre funciona, ilimitado)
    try:
        url = await gerar_imagem_pollinations(prompt_en)
        return {"ok": True, "imagem": url, "modelo": "🌸 Pollinations", "prompt_en": prompt_en}
    except Exception as e:
        erros_img.append(f"Pollinations: {e}")
        print(f"[Pollinations] falhou: {e}")

    # Fallback final — Pollinations URL direta
    import urllib.parse as _ul
    p = _ul.quote(prompt_en[:400])
    url = f"https://image.pollinations.ai/prompt/{p}?width=1280&height=1280&model=flux-pro&nologo=true&enhance=true"
    return {"ok": True, "imagem": url, "modelo": "🌸 Pollinations", "prompt_en": prompt_en}

    # Outros modelos específicos
    if modelo_req == "pollinations":
        import urllib.parse as _ul
        # Enriquecer prompt para máxima qualidade
        prompt_rich = prompt_en + ", masterpiece, ultra detailed, professional photography, 8k uhd, sharp focus, perfect composition, award winning, trending on artstation"
        p = _ul.quote(prompt_rich[:600])
        url = f"https://image.pollinations.ai/prompt/{p}?width=1280&height=1280&model=flux-pro&nologo=true&enhance=true&seed={hash(prompt_en)%99999}"
        return {"ok": True, "imagem": url, "modelo": "🌸 Pollinations Flux Pro", "prompt_en": prompt_rich}

    # Fallback automático — tenta tudo
    if hf_key:
        try:
            url = await gerar_imagem_hf(prompt_en)
            return {"ok": True, "imagem": url, "modelo": "🤗 HuggingFace FLUX (auto)", "prompt_en": prompt_en}
        except: pass

    try:
        url = await gerar_imagem_gemini(prompt_en)
        return {"ok": True, "imagem": url, "modelo": "🍌 Gemini (auto)", "prompt_en": prompt_en}
    except: pass

    # Raphael AI — FLUX.1 Dev sem key, ilimitado
    try:
        import urllib.parse as _ul
        p = _ul.quote(prompt_en[:500])
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            r = await client.get(f"https://raphael.app/api/generate?prompt={p}&model=flux-dev&width=1024&height=1024")
            if r.is_success:
                d = r.json()
                if d.get("url") or d.get("image_url"):
                    url = d.get("url") or d.get("image_url")
                    return {"ok": True, "imagem": url, "modelo": "✏️ Raphael AI FLUX Dev", "prompt_en": prompt_en}
    except Exception as e:
        print(f"[Raphael] falhou: {e}")

    # Fallback final — Pollinations
    import urllib.parse as _ul
    prompt_rich = prompt_en + ", masterpiece, ultra detailed, 8k uhd, sharp focus, cinematic"
    prompt_safe = _ul.quote(prompt_rich[:600])
    url = f"https://image.pollinations.ai/prompt/{prompt_safe}?width=1280&height=1280&model=flux-pro&nologo=true&enhance=true&seed={hash(prompt_en)%99999}"
    return {"ok": True, "imagem": url, "modelo": "🌸 Pollinations Flux Pro", "prompt_en": prompt_en}



# ══════════════════════════════════════════════════════════════
# VORTEX STUDIO — Pipeline completo de vídeo
# Roteiro → Voz → Imagem → Vídeo = Vídeo único do Vortex
# ══════════════════════════════════════════════════════════════

class VortexStudioRequest(BaseModel):
    tema:       str
    nicho:      str = "geral"
    estilo:     str = "cinematografico"  # cinematografico, viral, educacional
    voz_id:     str = "default"          # ID da voz clonada
    modelo_video: str = "kling3_std"     # modelo de vídeo a usar
    duracao:    int = 30                 # segundos

@app.post("/vortex-studio/criar")
@limiter.limit("3/minute")
async def vortex_studio(data: VortexStudioRequest, request: Request):
    """
    Vortex Studio — Pipeline completo:
    1. Claude Sonnet gera roteiro + prompt visual
    2. ElevenLabs gera narração
    3. FLUX gera imagem base com estilo Vortex
    4. Kling/WAN anima a imagem com a voz

    Resultado: vídeo único com identidade visual Vortex.
    """
    usuario_id = extrair_usuario_id(request)

    # Verificar plano — só Creator+
    from creditos import get_usuario
    user_data = get_usuario(usuario_id)
    plano = user_data.get("plano", "free")
    planos_permitidos = ["creator", "pro", "elite", "elite_lifetime"]
    if plano not in planos_permitidos:
        raise HTTPException(403, "Vortex Studio disponível a partir do plano Creator.")

    # Calcular custo total
    CUSTO_STUDIO = {
        "wan22_fast":  60,
        "kling3_std":  90,
        "kling3_pro":  150,
        "luma_ray3":   155,
        "veo31_fast":  185,
    }
    creditos_necessarios = CUSTO_STUDIO.get(data.modelo_video, 90)

    from creditos import verificar_saldo, debitar_creditos
    if verificar_saldo(usuario_id, creditos_necessarios) < creditos_necessarios:
        raise HTTPException(402, f"Créditos insuficientes. Vortex Studio requer {creditos_necessarios} créditos.")

    resultado = {
        "status": "processando",
        "etapas": {
            "roteiro": {"status": "pendente", "resultado": None},
            "voz":     {"status": "pendente", "resultado": None},
            "imagem":  {"status": "pendente", "resultado": None},
            "video":   {"status": "pendente", "resultado": None},
        },
        "video_final": None,
        "creditos_usados": creditos_necessarios,
    }

    # ESTILOS VISUAIS DO VORTEX
    ESTILOS_VORTEX = {
        "cinematografico": "cinematic dark atmosphere, dramatic lighting, film grain, 4K ultra realistic, moody shadows, professional color grading",
        "viral":           "vibrant colors, high contrast, trending aesthetic, eye-catching composition, social media optimized, bold visuals",
        "educacional":     "clean modern design, bright lighting, professional studio look, clear and engaging visuals",
        "terror":          "dark horror atmosphere, unsettling shadows, eerie fog, dramatic tension, psychological horror aesthetic",
        "gaming":          "neon RGB lighting, dynamic composition, gaming aesthetic, high energy, explosive visual effects",
    }
    estilo_visual = ESTILOS_VORTEX.get(data.estilo, ESTILOS_VORTEX["cinematografico"])

    try:
        # ── ETAPA 1: ROTEIRO com Claude ──────────────────────────
        resultado["etapas"]["roteiro"]["status"] = "processando"
        print(f"[STUDIO] Etapa 1/4 — Gerando roteiro com Claude...")

        ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        if not ANTHROPIC_KEY:
            raise Exception("ANTHROPIC_API_KEY não configurada")

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-5-20251001",
                    "max_tokens": 1000,
                    "messages": [{
                        "role": "user",
                        "content": f"""Crie um roteiro viral para TikTok sobre "{data.tema}" no nicho de {data.nicho}.

O roteiro deve ter:
1. NARRAÇÃO: texto para narrar (máx 150 palavras para {data.duracao}s)
2. PROMPT_VISUAL: descrição em inglês da cena principal para gerar imagem

Formato de resposta:
NARRAÇÃO: [texto aqui]
PROMPT_VISUAL: [descrição visual em inglês aqui]

Estilo visual: {data.estilo}"""
                    }],
                }
            )

        if not r.is_success:
            raise Exception(f"Claude falhou: {r.status_code}")

        claude_resp = r.json()["content"][0]["text"]
        naracao = ""
        prompt_visual = ""

        for linha in claude_resp.split("\n"):
            if linha.startswith("NARRAÇÃO:"):
                naracao = linha.replace("NARRAÇÃO:", "").strip()
            elif linha.startswith("PROMPT_VISUAL:"):
                prompt_visual = linha.replace("PROMPT_VISUAL:", "").strip()

        if not naracao:
            naracao = claude_resp[:300]
        if not prompt_visual:
            prompt_visual = f"{data.tema} {estilo_visual}"

        # Adicionar estilo Vortex ao prompt visual
        prompt_visual_final = f"{prompt_visual}, {estilo_visual}"

        resultado["etapas"]["roteiro"] = {
            "status": "concluido",
            "resultado": {"naracao": naracao, "prompt_visual": prompt_visual_final}
        }
        print(f"[STUDIO] ✅ Etapa 1 concluída — {len(naracao)} chars de narração")

        # ── ETAPA 2: VOZ com ElevenLabs ──────────────────────────
        resultado["etapas"]["voz"]["status"] = "processando"
        print(f"[STUDIO] Etapa 2/4 — Gerando voz com ElevenLabs...")

        EL_KEY = os.getenv("ELEVENLABS_API_KEY", "")
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel — voz padrão Vortex

        audio_url = None
        if EL_KEY:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                r_voz = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={"xi-api-key": EL_KEY, "Content-Type": "application/json"},
                    json={
                        "text": naracao,
                        "model_id": "eleven_flash_v2_5",
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
                    }
                )
                if r_voz.is_success:
                    import base64
                    audio_b64 = base64.b64encode(r_voz.content).decode()
                    audio_url = f"data:audio/mpeg;base64,{audio_b64[:100]}..."
                    print(f"[STUDIO] ✅ Etapa 2 concluída — áudio gerado")

        resultado["etapas"]["voz"] = {
            "status": "concluido" if audio_url else "fallback",
            "resultado": {"audio_gerado": bool(audio_url)}
        }

        # ── ETAPA 3: IMAGEM com FLUX ──────────────────────────────
        resultado["etapas"]["imagem"]["status"] = "processando"
        print(f"[STUDIO] Etapa 3/4 — Gerando imagem base...")

        FAL_KEY = os.getenv("FAL_API_KEY", "")
        imagem_url = None

        if FAL_KEY:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                r_img = await client.post(
                    "https://fal.run/fal-ai/flux/schnell",
                    headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
                    json={
                        "prompt": prompt_visual_final,
                        "image_size": "landscape_16_9",
                        "num_inference_steps": 4,
                        "num_images": 1,
                    }
                )
                if r_img.is_success:
                    img_data = r_img.json()
                    imagem_url = img_data.get("images", [{}])[0].get("url", "")
                    print(f"[STUDIO] ✅ Etapa 3 concluída — imagem gerada")

        # Fallback: Pollinations grátis
        if not imagem_url:
            import urllib.parse
            imagem_url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt_visual_final[:200])}"
            print(f"[STUDIO] ⚠️ Usando Pollinations como fallback")

        resultado["etapas"]["imagem"] = {
            "status": "concluido",
            "resultado": {"url": imagem_url}
        }

        # ── ETAPA 4: VÍDEO com FAL ────────────────────────────────
        resultado["etapas"]["video"]["status"] = "processando"
        print(f"[STUDIO] Etapa 4/4 — Animando com {data.modelo_video}...")

        MODELOS_FAL = {
            "wan22_fast":  "fal-ai/wan/t2v-1.3b",
            "kling3_std":  "fal-ai/kling-video/v1.6/standard/text-to-video",
            "kling3_pro":  "fal-ai/kling-video/v1.6/pro/text-to-video",
            "luma_ray3":   "fal-ai/luma-dream-machine/ray-3",
            "veo31_fast":  "fal-ai/veo3/fast",
        }
        fal_model = MODELOS_FAL.get(data.modelo_video, "fal-ai/wan/t2v-1.3b")

        video_url = None
        if FAL_KEY:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                r_vid = await client.post(
                    f"https://fal.run/{fal_model}",
                    headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
                    json={
                        "prompt": prompt_visual_final,
                        "image_url": imagem_url,
                        "duration": min(data.duracao, 10),
                    }
                )
                if r_vid.is_success:
                    vid_data = r_vid.json()
                    video_url = vid_data.get("video", {}).get("url", "")
                    print(f"[STUDIO] ✅ Etapa 4 concluída — vídeo gerado!")

        resultado["etapas"]["video"] = {
            "status": "concluido" if video_url else "erro",
            "resultado": {"url": video_url}
        }
        resultado["video_final"] = video_url
        resultado["status"] = "concluido"

        # Debitar créditos só se gerou vídeo
        if video_url:
            debitar_creditos(usuario_id, creditos_necessarios, "vortex_studio")
            # Atualizar DNA
            atualizar_dna_com_roteiro(usuario_id, naracao, 8.0, data.nicho)
            # Registrar dataset
            registrar_par_treinamento(data.tema, naracao, tipo="studio")

        return {
            "ok": bool(video_url),
            "resultado": resultado,
            "mensagem": "✅ Vídeo criado pelo Vortex Studio!" if video_url else "⚠️ Erro na geração do vídeo",
        }

    except Exception as e:
        print(f"[STUDIO] ❌ Erro: {e}")
        raise HTTPException(500, f"Vortex Studio falhou: {str(e)}")


# ══════════════════════════════════════════════════════════════
# GEMMA 4 VISION — análise de imagem grátis
# ══════════════════════════════════════════════════════════════
@app.post("/gemma4/analisar-imagem")
async def gemma4_analisar(data: dict, request: Request):
    """
    Analisa imagem com Gemma 4 — grátis com HF_API_KEY.
    Usos: analisar thumbnail, ver objetos, ler texto em imagem.
    """
    imagem_url = data.get("url", "")
    pergunta   = data.get("pergunta", "O que você vê nessa imagem? Seja detalhado.")

    if not imagem_url:
        raise HTTPException(400, "URL da imagem obrigatória")

    try:
        resultado = await gemma4_analisar_imagem(imagem_url, pergunta)
        return {"ok": True, "analise": resultado, "modelo": "gemma-4-27b-it"}
    except Exception as e:
        raise HTTPException(500, f"Gemma 4 Vision falhou: {str(e)}")


# ══════════════════════════════════════════════════════════════
# AIML — FEATURE 1: Imagem com FLUX/GPT Image via AIML
# ══════════════════════════════════════════════════════════════
@app.post("/aiml/gerar-imagem")
async def aiml_imagem(data: dict, request: Request = None):
    """Gera imagem via AIML — FLUX Schnell (grátis) ou GPT Image 1.5"""
    prompt  = data.get("prompt", "")
    modelo  = data.get("modelo", "flux/schnell")
    tamanho = data.get("tamanho", "1024x1024")
    
    if not prompt:
        raise HTTPException(400, "Prompt obrigatório")
    
    aiml_key = os.getenv("AIML_API_KEY", AIML_API_KEY)
    if not aiml_key:
        raise HTTPException(503, "AIML_API_KEY não configurada — adicione no Render")
    
    try:
        url = await aiml_gerar_imagem(prompt, modelo, tamanho)
        return {"ok": True, "url": url, "modelo": modelo, "provider": "aiml"}
    except Exception as e:
        raise HTTPException(500, f"AIML imagem falhou: {str(e)}")


# ══════════════════════════════════════════════════════════════
# AIML — FEATURE 2: Vídeo com Veo 3.1 / Kling / WAN via AIML
# ══════════════════════════════════════════════════════════════
@app.post("/aiml/gerar-video")
async def aiml_video(data: dict, request: Request = None):
    """
    Gera vídeo via AIML com polling automático.
    Modelos: google/veo-3.0-generate, kling-video/v1.5/standard/text-to-video
    """
    prompt     = data.get("prompt", "")
    modelo     = data.get("modelo", "google/veo-3.0-generate")
    imagem_url = data.get("imagem_url", "")
    
    if not prompt:
        raise HTTPException(400, "Prompt obrigatório")
    
    aiml_key = os.getenv("AIML_API_KEY", AIML_API_KEY)
    if not aiml_key:
        raise HTTPException(503, "AIML_API_KEY não configurada — adicione no Render")
    
    try:
        video_url = await aiml_gerar_video(prompt, modelo, imagem_url)
        return {"ok": True, "video_url": video_url, "modelo": modelo, "provider": "aiml"}
    except Exception as e:
        raise HTTPException(500, f"AIML vídeo falhou: {str(e)}")


# ══════════════════════════════════════════════════════════════
# AIML — FEATURE 3: TTS — Narrar roteiro automaticamente
# ══════════════════════════════════════════════════════════════
@app.post("/aiml/tts")
async def aiml_tts(data: dict, request: Request = None):
    """
    Converte texto em áudio via AIML (OpenAI TTS).
    Vozes: alloy, echo, fable, onyx, nova, shimmer
    Modelos: tts-1, tts-1-hd
    """
    import base64
    texto  = data.get("texto", "")
    voz    = data.get("voz", "nova")
    modelo = data.get("modelo", "tts-1")
    
    if not texto:
        raise HTTPException(400, "Texto obrigatório")
    
    aiml_key = os.getenv("AIML_API_KEY", AIML_API_KEY)
    if not aiml_key:
        raise HTTPException(503, "AIML_API_KEY não configurada — adicione no Render")
    
    try:
        audio_bytes = await aiml_text_to_speech(texto, voz, modelo)
        audio_b64   = base64.b64encode(audio_bytes).decode()
        return {
            "ok": True,
            "audio_url": f"data:audio/mp3;base64,{audio_b64}",
            "voz": voz,
            "modelo": modelo,
            "provider": "aiml",
            "duracao_estimada": f"~{len(texto)//15}s"
        }
    except Exception as e:
        raise HTTPException(500, f"AIML TTS falhou: {str(e)}")

@app.post("/gerar-video-free")
async def gerar_video_free(request: VideoRequest):
    """Vídeo gratuito via HuggingFace LTX-Video."""
    prompt_en = request.prompt + ", cinematic, high quality, smooth motion"
    hf_key = os.getenv("HF_API_KEY", "")
    if not hf_key:
        raise HTTPException(500, "HF_API_KEY não configurada. Configure no Render Environment.")
    try:
        url = await gerar_video_hf(prompt_en)
        return {"ok": True, "video_url": url, "modelo": "HuggingFace LTX-Video (free)", "prompt_en": prompt_en}
    except Exception as e:
        raise HTTPException(502, f"Vídeo HF falhou: {str(e)[:100]}")


@app.post("/cerebro/feedback")
async def cerebro_feedback(request: Request):
    """
    Recebe feedback do usuário sobre roteiros e imagens.
    Isso alimenta o aprendizado do cérebro do Vortex.
    """
    data = await request.json()
    tipo = data.get("tipo", "roteiro")
    aprovado = data.get("aprovado", False)
    conteudo = data.get("conteudo", "")
    nicho = data.get("nicho", _perfil.get("nicho", "geral"))
    score = data.get("score_viral", 0.0)
    
    if tipo == "roteiro":
        aprender_com_roteiro(conteudo, nicho, aprovado, score)
    elif tipo == "imagem":
        modelo = data.get("modelo", "")
        aprender_com_prompt_imagem(conteudo, modelo, aprovado)
    
    return {"ok": True, "mensagem": "Vortex aprendeu com seu feedback!"}


@app.get("/cerebro/status")
async def cerebro_status():
    """Retorna o estado atual do cérebro do Vortex."""
    estado = get_estado_cerebro()
    return {"ok": True, "cerebro": estado}


@app.get("/cerebro/insights/{nicho}")
async def cerebro_insights(nicho: str):
    """Retorna insights aprendidos sobre um nicho."""
    insights = get_insights_nicho(nicho)
    return {"ok": True, "insights": insights}


@app.get("/me")
def get_me(request: Request):
    usuario_id = extrair_usuario_id(request)
    saldo = get_saldo(usuario_id)
    hist = historico_creditos(usuario_id, 5)
    return {
        "ok": True,
        "saldo": saldo,
        "perfil": _perfil,
        "perfil_completo": perfil_completo(),
        "dna": bool(_dna_criador),
        "canais": len(_canais),
        "historico_recente": hist,
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "═"*70)
    print("  🔥 VORTEX AI Backend v6.0 HOLLYWOOD EDITION")
    print("  🌐 http://127.0.0.1:8082")
    print("═"*70)
    print(f"  Groq          {'✅' if GROQ_API_KEY        else '❌'}")
    print(f"  Gemini        {'✅' if GEMINI_API_KEY      else '❌'}")
    print(f"  Leonardo AI   {'✅' if LEONARDO_API_KEY    else '❌'}")
    print(f"  Runway Gen-3  {'✅' if RUNWAY_API_KEY      else '❌'}")
    print(f"  ElevenLabs    {'✅' if ELEVENLABS_API_KEY  else '❌'}")
    print("═"*70)
    print("  🎬 MODO DIRETOR: segundo a segundo")
    print("  📊 SCORE VIRAL: 5 dimensões")
    print("  🧬 DNA CRIADOR: aprende com roteiros aprovados")
    print("  📺 MODO SÉRIE: 3 episódios conectados")
    print("  🅰️🅱️ MODO A/B: 2 versões por roteiro")
    print("  📅 CALENDÁRIO: melhor dia/hora por plataforma")
    print("  🏢 MODO AGÊNCIA: múltiplos canais")
    print("═"*70+"\n")
    uvicorn.run(app, host="127.0.0.1", port=8082, reload=False)