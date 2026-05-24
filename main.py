import os, json, asyncio, base64
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
VORTEX_URL      = os.getenv("VORTEX_URL", "http://localhost:5173")

app = FastAPI(title="Vortex AI Backend", version="6.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
_limite = {"data": str(date.today()), "usado": 0, "limite": 100}
_feedbacks: list = []


# ══════════════════════════════════════════════════════════════
# 🎬 CLOUDO MODELO v6.0 — ROTEIRISTA DE HOLLYWOOD
# ══════════════════════════════════════════════════════════════

VORTEX_CHAT = """Você é o VORTEX — a IA mais completa e afiada do Brasil.

Você é um assistente GERAL e inteligente. Pode ajudar com QUALQUER coisa:
• Perguntas gerais, curiosidades, história, ciência, tecnologia
• Estratégia de conteúdo e crescimento nas redes sociais
• Marketing digital, copywriting, vendas
• Negócios, finanças, empreendedorismo
• Programação, tecnologia, IA
• Criatividade, ideias, brainstorming
• E muito mais — você sabe de tudo

Personalidade:
• Direto, confiante, inteligente e sem enrolação
• Fala de forma natural, como um amigo expert
• Responde de forma objetiva — vai direto ao ponto
• Usa exemplos práticos e reais
• Tem senso de humor quando apropriado

Especialidade extra — Conteúdo Viral:
• Expert em TikTok, Reels, YouTube Shorts e algoritmos
• Sabe o que está bombando e por quê
• Entende o mercado brasileiro de criadores profundamente
• Estratégias reais de crescimento por nicho

Regras importantes:
• NUNCA invente dados sobre pessoas reais — se não souber, diga claramente
• Respostas curtas e diretas para perguntas simples
• Respostas detalhadas apenas quando necessário
• Quando perguntarem sobre algo atual, use os dados disponíveis
• NUNCA gere roteiro completo no chat — para isso existe a aba Roteiro

Você é o assistente mais poderoso que um criador brasileiro pode ter."""

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

def checar_limite():
    hoje = str(date.today())
    if _limite["data"] != hoje:
        _limite["data"] = hoje
        _limite["usado"] = 0
    return _limite

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
async def auth_google():
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
        usuario = get_usuario_db(usuario_id)
        if not usuario.get("creditos"):
            usuario["creditos"] = 50  # créditos grátis para novo usuário
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
async def auth_logout():
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
async def onboarding(data: OnboardingIn):
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
        pass

    return {"ok": True, "perfil_completo": True, "perfil": _perfil, "boas_vindas": boas_vindas, "roteiro_exemplo": roteiro_exemplo}


# ══════════════════════════════════════════════════════════════
# ROTA — CHAT
# ══════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat(data: ChatRequest):
    lim = checar_limite()
    if lim["usado"] >= lim["limite"]:
        raise HTTPException(429, "Limite diário atingido.")
    usuario_id = "default"
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
    
    if plano in ["elite_lifetime", "elite_mensal", "elite_anual"]:
        config["provedor"] = "claude_sonnet" if anthropic_ok else "gemini"
        config["max_tokens"] = 4000
        print(f"[CLOUDO] ELITE — {'Claude Sonnet' if anthropic_ok else 'Gemini'} / 4000 tokens 🎬")
    elif plano in ["ultra_mensal", "ultra_anual", "pro_mensal", "pro_anual"]:
        config["provedor"] = "claude_haiku" if anthropic_ok else "gemini"
        config["max_tokens"] = 3000
        print(f"[CLOUDO] PRO — {'Claude Haiku' if anthropic_ok else 'Gemini'} / 3000 tokens")
    elif plano in ["creator_mensal", "creator_anual"]:
        config["provedor"] = "gemini"
        config["max_tokens"] = 2500
        print(f"[CLOUDO] CREATOR — Gemini / 2500 tokens")
    elif plano in ["starter_mensal", "starter_anual"]:
        config["provedor"] = "gemini"
        config["max_tokens"] = 2000
        print(f"[CLOUDO] STARTER — Gemini / 2000 tokens")
    else:
        # Free — Gemini estável
        config["provedor"] = "gemini"
        config["max_tokens"] = 1500
        print(f"[CLOUDO] FREE — Gemini / 1500 tokens")

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
    
    kw_busca_real = ["caso","crime","assassin","maníaco","maniaco","serial killer",
                     "true crime","desaparec","mistério","misterio","acidente",
                     "tragédia","tragedia","historia real","caso real","quem foi",
                     "o que aconteceu","brasil","sp","rj","mg","pr","rs","ba","ce"]
    
    precisa_busca = (
        tavily_key and
        len(msg_atual) > 10 and
        any(kw in msg_atual.lower() for kw in kw_busca_real) and
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
        if config["provedor"] == "openrouter_chat":
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
            pass

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
async def gerar_roteiro(data: RoteiroIn):
    usuario_id = "default"
    creditos_map = {"curto":1,"medio":2,"longo":3,"completo":5}
    creditos_necessarios = creditos_map.get(data.formato, 2)
    if data.modo in ["serie","ab","completo"]: creditos_necessarios *= 2

    saldo = verificar_saldo(usuario_id, creditos_necessarios)
    if saldo < creditos_necessarios:
        raise HTTPException(402, f"Créditos insuficientes. Precisa de {creditos_necessarios}.")

    system = montar_system_vortex(usar_cloudo=True, canal_id=data.canal_id or "default")
    MAX_TOKENS = {"curto":800,"medio":1500,"longo":2500,"completo":3000}.get(data.formato, 1500)
    provedor = "gemini" if data.formato in ["longo","completo"] or data.modo in ["serie","completo"] else "groq"
    contexto = montar_contexto_criador(data.canal_id or "default")
    plataforma = _perfil.get("plataformas",["TikTok"])[0] if _perfil.get("plataformas") else "TikTok"

    # Tavily: busca fatos reais sobre o tema
    fatos_reais = ""
    if os.getenv("TAVILY_API_KEY", TAVILY_API_KEY):
        fatos_reais = await buscar_tavily(data.tema + " caso real Brasil")
        if fatos_reais:
            print(f"[Tavily] Fatos reais para roteiro: {len(fatos_reais)} chars")

    # Monta prompt baseado no modo
    if data.modo == "diretor":
        prompt = f"""Crie um roteiro DIRETOR COMPLETO para: {data.tema}
Formato: {data.formato} | Plataforma: {plataforma}
{f"Criador: {contexto}" if contexto else ""}

Entregue OBRIGATORIAMENTE:
- Roteiro segundo a segundo (com timestamps [00:00])
- Direção de câmera para cada cena
- Emoção do espectador em cada momento
- Tipo de corte/transição
- Efeito sonoro exato
- Score viral nas 5 dimensões ao final"""

    elif data.modo == "serie":
        prompt = f"""Crie uma SÉRIE de 3 episódios sobre: {data.tema}
Plataforma: {plataforma}
{f"Criador: {contexto}" if contexto else ""}

Para cada episódio entregue:
- Hook individual impossível de ignorar
- Roteiro completo
- Como esse episódio conecta com o próximo
- Cliffhanger que obriga a ver o próximo

Os 3 episódios devem formar uma narrativa que vicia o espectador no canal inteiro."""

    elif data.modo == "ab":
        prompt = f"""Crie 2 VERSÕES COMPLETAMENTE DIFERENTES do mesmo roteiro para: {data.tema}
Plataforma: {plataforma}
{f"Criador: {contexto}" if contexto else ""}

VERSÃO A: Abordagem emocional/drama
VERSÃO B: Abordagem curiosidade/mistério

Para cada versão: hook, desenvolvimento, virada, CTA invisível.
Ao final, analise qual tem maior potencial viral e por quê."""

    elif data.modo == "completo":
        prompt = f"""Crie o PACOTE COMPLETO de conteúdo para: {data.tema}
Formato: {data.formato} | Plataforma: {plataforma}
{f"Criador: {contexto}" if contexto else ""}

Entregue TUDO:
1. 3 opções de título viral
2. Roteiro segundo a segundo com direção
3. Direção completa (câmera, cena, emoção, edição)
4. Trilha sonora (música + momentos exatos)
5. Prompts IA para cada cena principal
6. Legenda pronta com emojis
7. 15 hashtags estratégicas (5 grandes + 5 nicho + 5 tendência)
8. CTA invisível específico
9. Score viral nas 5 dimensões
10. Preview do próximo episódio"""

    else:
        # Buscar tendências reais do nicho antes de gerar
        tendencias_reais = ""
        if TAVILY_API_KEY:
            try:
                tendencias_reais = await buscar_tavily(f"tendências virais {data.nicho or data.tema} TikTok 2026 Brasil")
            except:
                pass

        prompt = f"""Crie um roteiro viral PROFISSIONAL de 60 segundos para {plataforma}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRIEFING DO ROTEIRO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEMA: {data.tema}
FORMATO: {data.formato}
PLATAFORMA: {plataforma}
NICHO: {data.nicho or "conteúdo viral"}
{f"PERFIL DO CRIADOR: {contexto}" if contexto else ""}
{f"TENDÊNCIAS REAIS DO NICHO:{chr(10)}{tendencias_reais[:600]}" if tendencias_reais else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIRETRIZES CRIATIVAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PESQUISA: Busque mentalmente casos reais, crimes documentados, experimentos ou fatos verificáveis sobre "{data.tema}"
2. ESPECIFICIDADE: Use nomes, datas, lugares e números reais ou inventados com precisão cirúrgica
3. TÉCNICA: Escolha pelo menos UMA: POV imersivo / Revelação progressiva / Depoimento em 1ª pessoa / Fato chocante + ficção
4. HOOK: Afirmação perturbadora nos primeiros 2 segundos — NUNCA começa com pergunta
5. VIRADA: ATO 3 deve ser IMPOSSÍVEL de prever — se der para adivinhar, reescreve
6. FINAL: Último segundo força comentário, compartilhamento ou próximo episódio
7. FALAS: Todas as falas do narrador devem ser REAIS e completas — sem "[falar sobre X]"
8. SCORE: Só entrega se a média for ≥ 8/10 — reescreve se for menor

Siga EXATAMENTE o formato do manual."""
    
    # Injeta fatos reais se Tavily encontrou algo
    if fatos_reais:
        prompt = f"""FATOS REAIS PESQUISADOS:
{fatos_reais[:1000]}

---
{prompt}

Use os fatos reais acima. Seja específico — nomes reais, datas reais, locais reais."""

    # Usa cascata dedicada para roteiro — DeepSeek V3 > Qwen3 235b > Llama 70b > DeepSeek R1 > Gemini
    roteiro, modelo = await gerar_texto_roteiro(
        [{"role":"user","content":prompt}], system=system,
        max_tokens=MAX_TOKENS,
    )
    debitar_creditos(usuario_id, creditos_necessarios, f"roteiro_{data.formato}_{data.modo}")
    return {"ok":True,"roteiro":roteiro,"tema":data.tema,"formato":data.formato,
            "modo":data.modo,"modelo":modelo,"cloudo_ativo":True,"creditos_debitados":creditos_necessarios}


# ══════════════════════════════════════════════════════════════
# ROTA — SCORE VIRAL
# ══════════════════════════════════════════════════════════════

@app.post("/score-viral")
async def score_viral(data: ScoreRequest):
    usuario_id = "default"
    saldo = verificar_saldo(usuario_id, 2)
    if saldo < 2: raise HTTPException(402, "Créditos insuficientes. Precisa de 2.")

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
async def aprender_dna(data: DNARequest):
    usuario_id = "default"
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
async def calendario():
    usuario_id = "default"
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
async def relatorio_semanal():
    usuario_id = "default"
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
async def trends_agora():
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
async def analisar_perfil(data: AnalisarPerfilIn):
    usuario_id = "default"
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
                    pass

            prompt_ia = f"""Você é o melhor estrategista de {rede} do Brasil.

DADOS DO PERFIL @{perfil}:
- Seguidores: {resultado.get('seguidores', 'N/A')}
- Engajamento: {resultado.get('engajamento', 'N/A')}
- Posts: {resultado.get('posts', 'N/A')}
- Bio: {resultado.get('bio', 'N/A')}
{f"- Contexto do criador: {contexto}" if contexto else ""}
{f"- Dados do mercado: {dados_mercado[:400]}" if dados_mercado else ""}

Faça uma análise CIRÚRGICA e entregue:

📊 DIAGNÓSTICO (o que está funcionando e o que está matando o crescimento)

🎯 SCORE DO PERFIL: X/100
- Consistência de conteúdo: X/10
- Qualidade visual: X/10  
- Engajamento: X/10
- Estratégia de nicho: X/10
- Potencial viral: X/10

🚀 3 AÇÕES IMEDIATAS (o que fazer nos próximos 7 dias)

💡 INSIGHT OCULTO (algo que ninguém está vendo mas que pode explodir o perfil)

⚠️ ERRO CRÍTICO (o maior erro que está impedindo o crescimento)"""

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
async def analisar_video(data: AnalisarVideoIn):
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
            pass

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
async def gerar_imagem(request: ImageRequest):
    usuario_id = "default"
    creditos_map = {
        "wavespeed-ai/flux-dev": 5,
        "wavespeed-ai/flux-dev-ultra-fast": 4,
        "wavespeed-ai/flux-schnell": 2,
        "PHOENIX": 6,
        "LEONARDO_SIGNATURE": 8,
        "LEONARDO_CREATIVE": 7,
        "stability-ultra": 7,
        "ideogram-v2": 6,
    }
    creditos = creditos_map.get(request.modelo or "wavespeed-ai/flux-dev", 5)
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
    try:
        prompt_en = await interpretar_prompt_inteligente(prompt_base, tipo=tipo_prompt)
    except Exception as e_prompt:
        print(f"[gerar-imagem] prompt inteligente falhou: {e_prompt} — usando prompt original")
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

    if not url:
        raise HTTPException(500, f"Todas as APIs de imagem falharam: {'; '.join(erros)}")

    debitar_creditos(usuario_id, creditos, "gerar_imagem")
    return {"ok":True,"imagem":url,"modelo":modelo_usado,"prompt_en":prompt_en,"creditos_debitados":creditos}


# ══════════════════════════════════════════════════════════════
# ROTA — GERAR VÍDEO
# ══════════════════════════════════════════════════════════════

@app.post("/gerar-video")
async def gerar_video(request: VideoRequest):
    usuario_id = "default"
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
    try:
        prompt_en = await interpretar_prompt_inteligente(prompt_base, tipo=tipo_prompt)
    except Exception as e_prompt:
        print(f"[gerar-imagem] prompt inteligente falhou: {e_prompt} — usando prompt original")
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
    elif WAVESPEED_API_KEY:
        modelo_ws = request.modelo if request.modelo and "wavespeed" in request.modelo else "wavespeed-ai/wan-2.2/t2v-480p"
        try:
            url = await gerar_video_wavespeed(prompt=prompt_en, duracao=request.duracao, modelo=modelo_ws)
            modelo_usado = "WaveSpeed " + modelo_ws.split("/")[-1]
        except Exception as e_ws:
            # Fallback para Kling se WaveSpeed falhar
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
async def clonar_voz(audio: UploadFile = File(...), nome: str = "Minha Voz"):
    """Clona a voz do usuário via ElevenLabs Instant Voice Cloning."""
    usuario_id = "default"
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
async def gerar_musica(request: MusicaRequest):
    usuario_id = "default"
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
async def gerar_voz(request: VoiceRequest):
    usuario_id = "default"
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
async def tendencias(nicho: str = "", plataforma: str = "", pais: str = "BR", idioma: str = "pt"):
    """
    Sistema global de tendências — busca tendências reais em tempo real.
    Detecta o país e idioma automaticamente para resultados localizados.
    """
    usuario_id = "default"
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
    if TAVILY_API_KEY:
        try:
            query = f"tendências virais {nicho_final} {plataforma_final} {ctx_pais['nome']} 2026"
            resultado_tavily = await buscar_tavily(query, max_results=5)
            
            if resultado_tavily:
                # Processar com IA para extrair trends estruturadas
                prompt_trends = f"""Analise essas informações sobre tendências de {nicho_final} no {ctx_pais['nome']}:

{resultado_tavily[:1500]}

Extraia exatamente 5 tendências virais ATUAIS em formato JSON:
[
  {{
    "titulo": "Nome da tendência",
    "descricao": "O que é e por que está viralizando",
    "como_usar": "Como o criador de {nicho_final} pode usar isso",
    "potencial": "alto/médio/baixo",
    "hashtags": ["#tag1", "#tag2", "#tag3"]
  }}
]

Responda APENAS com o JSON, sem explicações."""

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

    # Fallback para trends estáticas se Tavily falhar
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
async def gerar_avatar(request: AvatarRequest):
    usuario_id = "default"
    creditos = 20
    saldo = verificar_saldo(usuario_id, creditos)
    if saldo < creditos:
        raise HTTPException(402, f"Créditos insuficientes. Precisa de {creditos}.")
    url = await gerar_avatar_hedra(imagem_url=request.imagem_url, audio_url=request.audio_url)
    debitar_creditos(usuario_id, creditos, "gerar_avatar")
    return {"ok": True, "video_url": url, "creditos_debitados": creditos}


@app.get("/creditos/saldo")
def creditos_saldo():
    return {"ok":True,"saldo":get_saldo("default")}

@app.get("/creditos/historico")
def creditos_hist():
    return {"ok":True,"historico":historico_creditos("default")}


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
async def pagamento_sucesso(plano: str, usuario: str = "default", payment_id: str = ""):
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
    """Webhook do Mercado Pago para confirmação assíncrona"""
    data = await request.json()
    print(f"[MP Webhook] {data}")
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
                                creditos_atuais = get_creditos_db(usuario_id)
                                set_creditos_db(usuario_id, creditos_atuais + plano_cfg["creditos"])
                                print(f"[MP Webhook] ✅ Créditos adicionados: {usuario_id} +{plano_cfg['creditos']}")
    return {"ok": True}

# ══════════════════════════════════════════════════════════════
# ROTA — UPLOAD DE VÍDEO
# ══════════════════════════════════════════════════════════════

@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
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
async def editar_video(request: EditarVideoRequest):
    usuario_id = "default"
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
async def gerar_imagem_wavespeed(prompt: str, endpoint: str = "wavespeed-ai/flux-dev") -> str:
    if not WAVESPEED_API_KEY:
        raise HTTPException(500, "WAVESPEED_API_KEY não configurada")
    ep = endpoint or "wavespeed-ai/flux-dev"
    # Submete o job
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(
            f"https://api.wavespeed.ai/api/v3/{ep}",
            headers={"Authorization": f"Bearer {WAVESPEED_API_KEY}", "Content-Type": "application/json"},
            json={"prompt": prompt, "width": 1024, "height": 1024, "num_images": 1, "seed": -1},
        )
        if not r.is_success: raise HTTPException(502, f"WaveSpeed imagem erro {r.status_code}: {r.text[:300]}")
        d = r.json()
        request_id = d.get("data", {}).get("id")
        if not request_id: raise HTTPException(502, f"WaveSpeed sem request_id: {d}")
        print(f"[WaveSpeed] Imagem job: {request_id}")

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


@app.get("/me")
def get_me():
    usuario_id = "default"
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