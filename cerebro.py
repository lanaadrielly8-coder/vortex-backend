"""
╔══════════════════════════════════════════════════════════════════╗
║           VORTEX AI — CÉREBRO PROPRIETÁRIO v1.0                 ║
║   Memória persistente + Internet + Aprendizado contínuo         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, json, asyncio, hashlib, time
from datetime import datetime, timedelta
from typing import Optional
import httpx

# ══════════════════════════════════════════════════════════════════
# MEMÓRIA PERSISTENTE — banco de dados do cérebro
# ══════════════════════════════════════════════════════════════════

CEREBRO_FILE = "cerebro_vortex.json"

def _carregar_cerebro() -> dict:
    """Carrega o cérebro do disco ou cria novo."""
    try:
        if os.path.exists(CEREBRO_FILE):
            with open(CEREBRO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {
        "versao": "1.0",
        "criado_em": datetime.now().isoformat(),
        "atualizado_em": datetime.now().isoformat(),
        
        # Aprendizado por nicho
        "nichos": {},          # {"terror": {"hooks": [], "formatos": [], "virais": []}}
        
        # Tendências salvas
        "tendencias": {},      # {"2026-05-24": {"TikTok": [...], "YouTube": [...]}}
        
        # Prompts que funcionaram
        "prompts_aprovados": [],  # lista de prompts com score
        
        # Roteiros aprovados pelo usuário
        "roteiros_aprovados": [],  # últimos 50
        
        # Padrões virais descobertos
        "padroes_virais": {
            "hooks_eficazes": [],
            "formatos_top": [],
            "emocoes_que_engajam": [],
            "durações_ideais": {},
        },
        
        # Histórico de gerações
        "historico": [],       # últimas 200 ações
        
        # Conhecimento acumulado
        "conhecimento": {
            "ultima_busca_internet": None,
            "fatos_relevantes": [],
            "tendencias_semana": [],
        },
        
        # Estatísticas de uso
        "stats": {
            "total_roteiros": 0,
            "total_imagens": 0,
            "total_videos": 0,
            "roteiros_aprovados": 0,
            "taxa_aprovacao": 0.0,
        }
    }

def _salvar_cerebro(cerebro: dict):
    """Salva o cérebro no disco."""
    cerebro["atualizado_em"] = datetime.now().isoformat()
    with open(CEREBRO_FILE, 'w', encoding='utf-8') as f:
        json.dump(cerebro, f, ensure_ascii=False, indent=2)

# Instância global do cérebro
_cerebro = _carregar_cerebro()


# ══════════════════════════════════════════════════════════════════
# SISTEMA DE APRENDIZADO — aprende com cada interação
# ══════════════════════════════════════════════════════════════════

def aprender_com_roteiro(roteiro: str, nicho: str, aprovado: bool, score_viral: float = 0.0):
    """
    Aprende com cada roteiro gerado.
    Se aprovado → extrai padrões e salva como referência.
    """
    global _cerebro
    
    # Atualizar stats
    _cerebro["stats"]["total_roteiros"] += 1
    if aprovado:
        _cerebro["stats"]["roteiros_aprovados"] += 1
    
    total = _cerebro["stats"]["total_roteiros"]
    aprov = _cerebro["stats"]["roteiros_aprovados"]
    _cerebro["stats"]["taxa_aprovacao"] = round(aprov / total * 100, 1) if total > 0 else 0.0
    
    # Inicializar nicho se necessário
    if nicho not in _cerebro["nichos"]:
        _cerebro["nichos"][nicho] = {
            "hooks": [],
            "formatos": [],
            "palavras_chave": [],
            "score_medio": 0.0,
            "total_gerados": 0,
            "total_aprovados": 0,
        }
    
    _cerebro["nichos"][nicho]["total_gerados"] += 1
    
    if aprovado:
        _cerebro["nichos"][nicho]["total_aprovados"] += 1
        
        # Salvar roteiro aprovado (máx 50)
        entrada = {
            "roteiro": roteiro[:500],  # primeiros 500 chars
            "nicho": nicho,
            "score": score_viral,
            "data": datetime.now().isoformat(),
        }
        _cerebro["roteiros_aprovados"].insert(0, entrada)
        _cerebro["roteiros_aprovados"] = _cerebro["roteiros_aprovados"][:50]
        
        # Extrair hook (primeira linha)
        linhas = roteiro.strip().split('\n')
        if linhas:
            hook = linhas[0][:200]
            if hook not in _cerebro["nichos"][nicho]["hooks"]:
                _cerebro["nichos"][nicho]["hooks"].insert(0, hook)
                _cerebro["nichos"][nicho]["hooks"] = _cerebro["nichos"][nicho]["hooks"][:20]
            
            # Também salvar nos padrões globais
            if hook not in _cerebro["padroes_virais"]["hooks_eficazes"]:
                _cerebro["padroes_virais"]["hooks_eficazes"].insert(0, hook)
                _cerebro["padroes_virais"]["hooks_eficazes"] = _cerebro["padroes_virais"]["hooks_eficazes"][:30]
    
    # Registrar no histórico
    _cerebro["historico"].insert(0, {
        "tipo": "roteiro",
        "nicho": nicho,
        "aprovado": aprovado,
        "score": score_viral,
        "data": datetime.now().isoformat(),
    })
    _cerebro["historico"] = _cerebro["historico"][:200]
    
    _salvar_cerebro(_cerebro)
    print(f"[CÉREBRO] Aprendeu com roteiro — nicho={nicho} aprovado={aprovado} taxa={_cerebro['stats']['taxa_aprovacao']}%")


def aprender_com_prompt_imagem(prompt: str, modelo: str, aprovado: bool):
    """Aprende quais prompts de imagem funcionam."""
    global _cerebro
    
    _cerebro["stats"]["total_imagens"] += 1
    
    if aprovado:
        entrada = {
            "prompt": prompt[:300],
            "modelo": modelo,
            "data": datetime.now().isoformat(),
        }
        _cerebro["prompts_aprovados"].insert(0, entrada)
        _cerebro["prompts_aprovados"] = _cerebro["prompts_aprovados"][:100]
    
    _salvar_cerebro(_cerebro)


def registrar_tendencia(plataforma: str, tendencias: list, nicho: str = ""):
    """Salva tendências descobertas."""
    global _cerebro
    
    hoje = datetime.now().strftime("%Y-%m-%d")
    if hoje not in _cerebro["tendencias"]:
        _cerebro["tendencias"][hoje] = {}
    
    key = f"{plataforma}_{nicho}" if nicho else plataforma
    _cerebro["tendencias"][hoje][key] = {
        "tendencias": tendencias[:10],
        "hora": datetime.now().strftime("%H:%M"),
    }
    
    # Manter só últimos 7 dias
    datas = sorted(_cerebro["tendencias"].keys(), reverse=True)
    for data_antiga in datas[7:]:
        del _cerebro["tendencias"][data_antiga]
    
    _cerebro["conhecimento"]["tendencias_semana"] = tendencias[:5]
    _salvar_cerebro(_cerebro)


# ══════════════════════════════════════════════════════════════════
# INTELIGÊNCIA CONECTADA — busca na internet antes de criar
# ══════════════════════════════════════════════════════════════════

_cache_internet = {}  # cache para evitar buscas repetidas

async def buscar_contexto_internet(tema: str, nicho: str = "", forcar: bool = False) -> str:
    """
    Busca contexto relevante na internet para enriquecer roteiros.
    Cache de 2 horas para não sobrecarregar a API.
    """
    cache_key = f"{tema}_{nicho}"
    agora = time.time()
    
    # Verificar cache
    if not forcar and cache_key in _cache_internet:
        cached = _cache_internet[cache_key]
        if agora - cached["ts"] < 7200:  # 2 horas
            print(f"[CÉREBRO] Cache hit: {cache_key[:50]}")
            return cached["resultado"]
    
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key:
        return ""
    
    try:
        # Montar query inteligente
        queries = []
        if nicho:
            queries.append(f"{tema} {nicho} viral 2026 tendência")
            queries.append(f"como fazer conteúdo {nicho} sobre {tema}")
        else:
            queries.append(f"{tema} viral trending 2026")
        
        resultados = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            for query in queries[:2]:  # máx 2 buscas
                r = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Content-Type": "application/json"},
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "search_depth": "basic",
                        "max_results": 3,
                        "include_answer": True,
                        "language": "pt",
                    }
                )
                if r.is_success:
                    d = r.json()
                    if d.get("answer"):
                        resultados.append(d["answer"])
                    for res in d.get("results", [])[:2]:
                        if res.get("content"):
                            resultados.append(f"[{res.get('title', '')}]: {res['content'][:300]}")
        
        contexto = "\n\n".join(resultados[:5])
        
        # Salvar no cache
        _cache_internet[cache_key] = {"resultado": contexto, "ts": agora}
        
        # Salvar no cérebro
        global _cerebro
        _cerebro["conhecimento"]["ultima_busca_internet"] = datetime.now().isoformat()
        if contexto:
            _cerebro["conhecimento"]["fatos_relevantes"].insert(0, {
                "tema": tema,
                "contexto": contexto[:200],
                "data": datetime.now().isoformat(),
            })
            _cerebro["conhecimento"]["fatos_relevantes"] = _cerebro["conhecimento"]["fatos_relevantes"][:20]
            _salvar_cerebro(_cerebro)
        
        print(f"[CÉREBRO] Internet: {len(contexto)} chars sobre '{tema}'")
        return contexto
        
    except Exception as e:
        print(f"[CÉREBRO] Busca internet falhou: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# CONSTRUTOR DE CONTEXTO — monta o contexto ideal para cada geração
# ══════════════════════════════════════════════════════════════════

async def construir_contexto_roteiro(
    tema: str,
    nicho: str,
    plataforma: str,
    buscar_net: bool = True,
) -> str:
    """
    Constrói o contexto completo para gerar um roteiro incrível.
    Combina: memória do nicho + tendências + internet + padrões virais.
    """
    global _cerebro
    partes = []
    
    # 1. Contexto do nicho aprendido
    if nicho in _cerebro["nichos"]:
        dados_nicho = _cerebro["nichos"][nicho]
        taxa = round(dados_nicho["total_aprovados"] / max(dados_nicho["total_gerados"], 1) * 100)
        partes.append(f"[MEMÓRIA DO NICHO {nicho.upper()}]")
        partes.append(f"Roteiros gerados: {dados_nicho['total_gerados']} | Taxa aprovação: {taxa}%")
        
        if dados_nicho["hooks"]:
            partes.append(f"Hooks que funcionaram: {' | '.join(dados_nicho['hooks'][:3])}")
    
    # 2. Tendências recentes
    hoje = datetime.now().strftime("%Y-%m-%d")
    ontem = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    for data in [hoje, ontem]:
        if data in _cerebro["tendencias"]:
            tends = _cerebro["tendencias"][data]
            key = f"{plataforma}_{nicho}"
            if key in tends:
                t_lista = tends[key].get("tendencias", [])
                if t_lista:
                    nomes = [t.get("titulo", str(t)) if isinstance(t, dict) else str(t) for t in t_lista[:3]]
                    partes.append(f"[TENDÊNCIAS {plataforma.upper()} HOJE]: {', '.join(nomes)}")
                    break
    
    # 3. Padrões virais globais
    if _cerebro["padroes_virais"]["hooks_eficazes"]:
        hooks_top = _cerebro["padroes_virais"]["hooks_eficazes"][:2]
        partes.append(f"[PADRÕES VIRAIS COMPROVADOS]: {' | '.join(hooks_top)}")
    
    # 4. Busca na internet (fatos reais)
    if buscar_net and tema:
        contexto_net = await buscar_contexto_internet(tema, nicho)
        if contexto_net:
            partes.append(f"[CONTEXTO REAL DA INTERNET]:\n{contexto_net[:600]}")
    
    # 5. Roteiros aprovados como referência
    roteiros_ref = [r for r in _cerebro["roteiros_aprovados"] if r.get("nicho") == nicho][:2]
    if roteiros_ref:
        partes.append(f"[REFERÊNCIA — roteiros aprovados anteriormente para {nicho}]:")
        for r in roteiros_ref:
            partes.append(f"  Score {r.get('score', 0):.1f}: {r['roteiro'][:150]}...")
    
    resultado = "\n".join(partes) if partes else ""
    
    if resultado:
        print(f"[CÉREBRO] Contexto construído: {len(resultado)} chars para tema='{tema}' nicho='{nicho}'")
    
    return resultado


async def construir_prompt_imagem_inteligente(
    descricao: str,
    estilo: str = "",
    nicho: str = "",
) -> str:
    """
    Constrói prompt de imagem usando IA + memória de prompts aprovados.
    Traduz PT→EN e enriquece baseado no que já funcionou.
    """
    global _cerebro
    
    # Buscar prompts similares aprovados
    prompts_ref = []
    for p in _cerebro["prompts_aprovados"][:10]:
        if nicho and nicho.lower() in p.get("prompt", "").lower():
            prompts_ref.append(p["prompt"])
    
    # Mapa de estilos
    ESTILOS = {
        "realista": "photorealistic, professional photography, natural lighting, sharp focus, 8k",
        "cinematico": "cinematic shot, anamorphic lens, dramatic lighting, movie scene, film grain",
        "dark": "dark atmosphere, moody gothic, chiaroscuro lighting, deep shadows, sinister",
        "anime": "anime style, vibrant colors, manga art, Studio Ghibli inspired, detailed",
        "3d": "3D render, octane render, volumetric lighting, subsurface scattering, CGI",
        "cartoon": "cartoon style, bold outlines, flat colors, comic book art, illustrated",
    }
    
    estilo_suffix = ESTILOS.get(estilo, "photorealistic, highly detailed, professional")
    
    # Usar IA para traduzir e enriquecer
    try:
        from providers import gerar_texto
        
        ref_text = ""
        if prompts_ref:
            ref_text = f"\nReferências de prompts aprovados anteriormente:\n" + "\n".join(prompts_ref[:2])
        
        system = f"""You are an expert AI image prompt engineer for {nicho or 'viral content'} creators.
Convert the user description to a detailed English prompt for AI image generation.

Style to use: {estilo_suffix}

Rules:
- Output ONLY the English prompt, nothing else
- Be specific: subject, lighting, composition, mood, camera angle
- Add: sharp focus, high detail, no blur, no motion blur, no artifacts, no watermark
- For thumbnails: make it eye-catching, high contrast, dramatic
- Maximum 100 words
{ref_text}"""

        msgs = [{"role": "user", "content": f"Create image prompt for: {descricao}"}]
        prompt_en, _ = await gerar_texto(msgs, system=system, max_tokens=120, provedor_preferido="groq")
        prompt_en = prompt_en.strip().strip('"').strip("'")
        
        # Adicionar sufixos de qualidade
        resultado = f"{prompt_en}, masterpiece, ultra detailed, 8k uhd, {estilo_suffix}, no blur, no artifacts"
        print(f"[CÉREBRO] Prompt imagem: {resultado[:80]}...")
        return resultado
        
    except Exception as e:
        print(f"[CÉREBRO] Prompt IA falhou: {e} — usando direto")
        return f"{descricao}, {estilo_suffix}, masterpiece, ultra detailed, 8k uhd, sharp focus, no blur"


# ══════════════════════════════════════════════════════════════════
# SELETOR DE IA POR PLANO — cada plano usa a melhor IA disponível
# ══════════════════════════════════════════════════════════════════

def selecionar_ia_roteiro(plano: str = "free") -> dict:
    """
    Seleciona a melhor IA para roteiro baseado no plano do usuário.
    Pro → Claude Opus | Creator → Gemini Pro | Starter → Kimi K2 | Free → DeepSeek
    """
    AIML_KEY = os.getenv("AIML_API_KEY", "")
    GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
    
    if plano == "pro" and AIML_KEY:
        return {
            "provedor": "aiml",
            "modelo": "claude-opus-4-5",
            "nome": "Claude Opus 4.6",
            "descricao": "Melhor para roteiros criativos e narrativas profundas",
        }
    elif plano in ["creator", "pro"] and AIML_KEY:
        return {
            "provedor": "aiml",
            "modelo": "gpt-4o",
            "nome": "GPT-4o",
            "descricao": "Excelente para roteiros estruturados e virais",
        }
    elif GEMINI_KEY:
        return {
            "provedor": "gemini",
            "modelo": "gemini-2.0-flash",
            "nome": "Gemini 2.0 Flash",
            "descricao": "Rápido e criativo para roteiros",
        }
    else:
        return {
            "provedor": "openrouter",
            "modelo": "deepseek/deepseek-chat-v3-0324:free",
            "nome": "DeepSeek V3",
            "descricao": "Gratuito e eficiente",
        }


async def chamar_ia_com_plano(
    messages: list,
    system: str,
    plano: str = "free",
    max_tokens: int = 2000,
) -> tuple[str, str]:
    """
    Chama a IA certa baseado no plano.
    Sempre retorna (texto, modelo_usado).
    """
    ia = selecionar_ia_roteiro(plano)
    
    AIML_KEY = os.getenv("AIML_API_KEY", "")
    
    try:
        if ia["provedor"] == "aiml" and AIML_KEY:
            # AIML API — acesso a Claude, GPT, etc
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.extend(messages)
            
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                r = await client.post(
                    "https://api.aimlapi.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {AIML_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": ia["modelo"],
                        "messages": msgs,
                        "max_tokens": max_tokens,
                        "temperature": 0.8,
                    }
                )
                if r.is_success:
                    d = r.json()
                    texto = d["choices"][0]["message"]["content"]
                    return texto, ia["nome"]
                    
        # Fallback para providers padrão
        from providers import gerar_texto_roteiro
        texto, modelo = await gerar_texto_roteiro(messages, system=system, max_tokens=max_tokens)
        return texto, modelo
        
    except Exception as e:
        print(f"[CÉREBRO] IA {ia['nome']} falhou: {e} — fallback")
        from providers import gerar_texto_roteiro
        texto, modelo = await gerar_texto_roteiro(messages, system=system, max_tokens=max_tokens)
        return texto, modelo


# ══════════════════════════════════════════════════════════════════
# API — endpoints para o frontend
# ══════════════════════════════════════════════════════════════════

def get_estado_cerebro() -> dict:
    """Retorna o estado atual do cérebro para o /status."""
    global _cerebro
    return {
        "versao": _cerebro.get("versao", "1.0"),
        "atualizado_em": _cerebro.get("atualizado_em"),
        "nichos_aprendidos": list(_cerebro["nichos"].keys()),
        "total_roteiros": _cerebro["stats"]["total_roteiros"],
        "taxa_aprovacao": _cerebro["stats"]["taxa_aprovacao"],
        "hooks_virais": len(_cerebro["padroes_virais"]["hooks_eficazes"]),
        "ultima_busca_internet": _cerebro["conhecimento"]["ultima_busca_internet"],
        "tendencias_hoje": len(_cerebro["tendencias"].get(datetime.now().strftime("%Y-%m-%d"), {})),
    }


def get_insights_nicho(nicho: str) -> dict:
    """Retorna insights aprendidos sobre um nicho específico."""
    global _cerebro
    
    if nicho not in _cerebro["nichos"]:
        return {"nicho": nicho, "dados": "Nenhum dado ainda — gere roteiros para aprender!"}
    
    dados = _cerebro["nichos"][nicho]
    total = dados["total_gerados"]
    aprov = dados["total_aprovados"]
    
    return {
        "nicho": nicho,
        "total_gerados": total,
        "total_aprovados": aprov,
        "taxa_aprovacao": round(aprov / max(total, 1) * 100, 1),
        "hooks_top": dados["hooks"][:5],
        "dica": f"Você já gerou {total} roteiros de {nicho}. Os melhores hooks estão salvos!"
    }



# ══════════════════════════════════════════════════════════════
# DNA DO CRIADOR — perfil que evolui com cada interação
# ══════════════════════════════════════════════════════════════

def salvar_dna(usuario_id: str, dados: dict):
    """
    Salva o DNA do criador — o perfil que o Vortex aprende ao longo do tempo.
    Cada interação enriquece o DNA.
    """
    global _cerebro
    if "dna_criadores" not in _cerebro:
        _cerebro["dna_criadores"] = {}

    if usuario_id not in _cerebro["dna_criadores"]:
        _cerebro["dna_criadores"][usuario_id] = {
            "nicho":            "",
            "estilo_escrita":   "",
            "tom":              "",
            "plataformas":      [],
            "hooks_favoritos":  [],
            "palavras_evitar":  [],
            "score_medio":      0.0,
            "total_roteiros":   0,
            "roteiros_virais":  [],
            "ultima_atualizacao": "",
        }

    dna = _cerebro["dna_criadores"][usuario_id]
    dna.update({k: v for k, v in dados.items() if v})
    dna["ultima_atualizacao"] = datetime.now().isoformat()

    _salvar_cerebro(_cerebro)
    print(f"[DNA] Atualizado para {usuario_id}: {list(dados.keys())}")


def carregar_dna(usuario_id: str) -> dict:
    """Carrega o DNA do criador."""
    global _cerebro
    if "dna_criadores" not in _cerebro:
        return {}
    return _cerebro["dna_criadores"].get(usuario_id, {})


def atualizar_dna_com_roteiro(usuario_id: str, roteiro: str, score: float, nicho: str):
    """
    Atualiza o DNA automaticamente após cada roteiro gerado.
    O Vortex aprende o estilo do criador com o tempo.
    """
    dna = carregar_dna(usuario_id)

    # Incrementar contador
    total = dna.get("total_roteiros", 0) + 1
    score_medio_atual = dna.get("score_medio", 0.0)
    score_medio_novo = round((score_medio_atual * (total-1) + score) / total, 2)

    # Extrair hook da primeira linha
    hook = roteiro.strip().split("\n")[0][:200] if roteiro else ""

    # Guardar roteiros virais (score > 8.0)
    roteiros_virais = dna.get("roteiros_virais", [])
    if score >= 8.0:
        roteiros_virais.insert(0, {
            "hook": hook,
            "score": score,
            "nicho": nicho,
            "data": datetime.now().isoformat(),
        })
        roteiros_virais = roteiros_virais[:20]  # Guarda os 20 melhores

    salvar_dna(usuario_id, {
        "nicho": nicho,
        "score_medio": score_medio_novo,
        "total_roteiros": total,
        "roteiros_virais": roteiros_virais,
    })
    print(f"[DNA] Score médio de {usuario_id}: {score_medio_novo} ({total} roteiros)")


# ══════════════════════════════════════════════════════════════
# SISTEMA DE FEEDBACK — o Vortex aprende o que funciona
# ══════════════════════════════════════════════════════════════

def salvar_feedback(usuario_id: str, tipo: str, conteudo: str, avaliacao: int, contexto: dict = {}):
    """
    Salva feedback do usuário sobre qualquer interação.
    O Vortex usa isso para melhorar com o tempo.

    tipo: 'roteiro' | 'chat' | 'imagem' | 'video' | 'tendencia'
    avaliacao: 1-5 (1=ruim, 5=excelente)
    """
    global _cerebro
    if "feedbacks" not in _cerebro:
        _cerebro["feedbacks"] = []

    feedback = {
        "usuario_id":  usuario_id,
        "tipo":        tipo,
        "conteudo":    conteudo[:300],
        "avaliacao":   avaliacao,
        "contexto":    contexto,
        "data":        datetime.now().isoformat(),
    }

    _cerebro["feedbacks"].insert(0, feedback)
    _cerebro["feedbacks"] = _cerebro["feedbacks"][:500]  # Guarda últimos 500

    # Aprender com feedback negativo
    if avaliacao <= 2 and tipo == "roteiro":
        if "padroes_evitar" not in _cerebro:
            _cerebro["padroes_evitar"] = []
        _cerebro["padroes_evitar"].insert(0, {
            "conteudo": conteudo[:200],
            "motivo": contexto.get("motivo", "avaliação baixa"),
            "data": datetime.now().isoformat(),
        })
        _cerebro["padroes_evitar"] = _cerebro["padroes_evitar"][:100]

    # Aprender com feedback positivo
    if avaliacao >= 4 and tipo == "roteiro":
        hook = conteudo.strip().split("\n")[0][:200]
        if hook and hook not in _cerebro["padroes_virais"]["hooks_eficazes"]:
            _cerebro["padroes_virais"]["hooks_eficazes"].insert(0, hook)
            _cerebro["padroes_virais"]["hooks_eficazes"] = _cerebro["padroes_virais"]["hooks_eficazes"][:30]

    _salvar_cerebro(_cerebro)
    print(f"[FEEDBACK] {tipo} avaliação={avaliacao}/5 de {usuario_id}")
    return feedback


def get_feedback_stats() -> dict:
    """Estatísticas dos feedbacks para análise."""
    global _cerebro
    feedbacks = _cerebro.get("feedbacks", [])
    if not feedbacks:
        return {"total": 0, "media": 0}

    por_tipo = {}
    for f in feedbacks:
        t = f["tipo"]
        if t not in por_tipo:
            por_tipo[t] = {"total": 0, "soma": 0}
        por_tipo[t]["total"] += 1
        por_tipo[t]["soma"] += f["avaliacao"]

    return {
        "total": len(feedbacks),
        "media_geral": round(sum(f["avaliacao"] for f in feedbacks) / len(feedbacks), 2),
        "por_tipo": {
            t: {
                "total": v["total"],
                "media": round(v["soma"] / v["total"], 2)
            }
            for t, v in por_tipo.items()
        }
    }


# ══════════════════════════════════════════════════════════════
# DATASET DE TREINAMENTO — coleta dados para Vortex Model v1
# ══════════════════════════════════════════════════════════════

def registrar_par_treinamento(pergunta: str, resposta: str, avaliacao: int = 0, tipo: str = "chat"):
    """
    Registra pares pergunta/resposta para treinamento futuro.
    Quando tiver 50.000 pares → treinar Vortex Model v1.
    """
    global _cerebro
    if "dataset_treinamento" not in _cerebro:
        _cerebro["dataset_treinamento"] = {
            "total_pares": 0,
            "meta": 50000,
            "pares": []
        }

    par = {
        "pergunta":  pergunta[:500],
        "resposta":  resposta[:1000],
        "avaliacao": avaliacao,
        "tipo":      tipo,
        "data":      datetime.now().isoformat(),
    }

    # Só guarda os melhores (avaliação alta ou sem avaliação ainda)
    if avaliacao >= 4 or avaliacao == 0:
        _cerebro["dataset_treinamento"]["pares"].insert(0, par)
        _cerebro["dataset_treinamento"]["pares"] = _cerebro["dataset_treinamento"]["pares"][:10000]

    _cerebro["dataset_treinamento"]["total_pares"] += 1
    total = _cerebro["dataset_treinamento"]["total_pares"]
    meta = _cerebro["dataset_treinamento"]["meta"]
    progresso = round(total / meta * 100, 1)

    _salvar_cerebro(_cerebro)

    if total % 1000 == 0:
        print(f"[DATASET] {total}/{meta} pares ({progresso}%) — Vortex Model v1 em construção!")

    return {"total": total, "progresso": progresso}