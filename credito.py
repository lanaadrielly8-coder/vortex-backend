"""
VORTEX AI — Sistema de Créditos v5.2
✅ Persistência em JSON — perfil e créditos salvos em disco
✅ Planos: Free, Pro, Ultra, Elite Lifetime
✅ Proteção anti-prejuízo + Rate limits
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional

# ── Arquivo de persistência ───────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "vortex_data.json")

# ── Cotação ───────────────────────────────────────────────────
USD_TO_BRL = 5.20

PLANOS = {
    "free": {
        "nome": "Free", "creditos": 5000, "preco_usd": 0, "preco_brl": 0,
        "limites_diarios": {"chat":10,"roteiro":3,"analise":1,"imagem":2,"video":0,"voz":2,"tendencias":2},
        "recursos": ["20 créditos grátis","Chat básico","Roteiros curtos","2 imagens/dia","Sem vídeos"],
    },
    "pro_mensal": {
        "nome": "Pro", "periodo": "Mensal", "creditos": 5000, "bonus": 50,
        "preco_usd": 9.99, "preco_brl": round(9.99*USD_TO_BRL,2),
        "limites_diarios": {"chat":50,"roteiro":15,"analise":5,"imagem":10,"video":2,"voz":10,"tendencias":10},
        "recursos": ["500 + 50 créditos bônus","CLOUDO MODELO","10 imagens/dia","2 vídeos/dia","Suporte prioritário"],
    },
    "pro_anual": {
        "nome": "Pro", "periodo": "Anual", "creditos": 6500, "bonus": 500,
        "preco_usd": 99.99, "preco_brl": round(99.99*USD_TO_BRL,2),
        "limites_diarios": {"chat":50,"roteiro":15,"analise":5,"imagem":10,"video":2,"voz":10,"tendencias":10},
        "recursos": ["6.500 créditos (+500 bônus)","2 meses grátis","Todos recursos Pro"],
    },
    "ultra_mensal": {
        "nome": "Ultra", "periodo": "Mensal", "creditos": 2000, "bonus": 300,
        "preco_usd": 29.99, "preco_brl": round(29.99*USD_TO_BRL,2),
        "limites_diarios": {"chat":-1,"roteiro":-1,"analise":20,"imagem":30,"video":10,"voz":30,"tendencias":-1},
        "recursos": ["2.000 + 300 créditos bônus","Tudo ilimitado","30 imagens/dia","10 vídeos/dia","API access"],
    },
    "ultra_anual": {
        "nome": "Ultra", "periodo": "Anual", "creditos": 26000, "bonus": 2000,
        "preco_usd": 299.99, "preco_brl": round(299.99*USD_TO_BRL,2),
        "limites_diarios": {"chat":-1,"roteiro":-1,"analise":20,"imagem":30,"video":10,"voz":30,"tendencias":-1},
        "recursos": ["26.000 créditos (+2.000 bônus)","3 meses grátis","Todos recursos Ultra"],
    },
    "elite_lifetime": {
        "nome": "Elite", "periodo": "Lifetime", "creditos": 100000, "bonus": 10000,
        "preco_usd": 999.99, "preco_brl": round(999.99*USD_TO_BRL,2),
        "recarga_mensal": 5000,
        "limites_diarios": {"chat":-1,"roteiro":-1,"analise":-1,"imagem":-1,"video":-1,"voz":-1,"tendencias":-1},
        "recursos": ["100.000 créditos iniciais","+5.000/mês pra sempre","SEM limites","Acesso vitalício"],
    },
}

PRECOS = {
    "chat":1,"roteiro_curto":2,"roteiro_medio":3,"roteiro_longo":5,
    "analise_perfil":4,"tendencias":4,"gerar_imagem":8,"gerar_video":15,"gerar_voz":4,
}

CUSTOS_REAIS_USD = {
    "chat":0,"roteiro":0,"analise_perfil":0.10,"tendencias":0.15,
    "gerar_imagem":0.07,"gerar_video":0.50,"gerar_voz":0.02,
}


# ══════════════════════════════════════════════════════════════
# PERSISTÊNCIA EM JSON
# ══════════════════════════════════════════════════════════════

def _load_data() -> dict:
    """Carrega dados do disco."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[DATA] Erro ao carregar: {e}")
    return {"usuarios": {}, "historico": {}, "perfil": {}, "dna": {}, "canais": {}}

def _save_data(data: dict):
    """Salva dados no disco."""
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"[DATA] Erro ao salvar: {e}")

# Carrega dados na inicialização
_data = _load_data()
_usuarios: dict = _data.get("usuarios", {})
_historico: dict = _data.get("historico", {})
_rate_limits: dict = {}

# ── Perfil persistente ────────────────────────────────────────
def salvar_perfil(perfil: dict):
    _data["perfil"] = perfil
    _save_data(_data)

def carregar_perfil() -> dict:
    return _data.get("perfil", {})

def salvar_dna(dna: dict):
    _data["dna"] = dna
    _save_data(_data)

def carregar_dna() -> dict:
    return _data.get("dna", {})

def salvar_canais(canais: dict):
    _data["canais"] = canais
    _save_data(_data)

def carregar_canais() -> dict:
    return _data.get("canais", {})


# ══════════════════════════════════════════════════════════════
# USUÁRIOS E SALDO
# ══════════════════════════════════════════════════════════════

def get_usuario(usuario_id: str) -> dict:
    if usuario_id not in _usuarios:
        _usuarios[usuario_id] = {
            "plano": "free",
            "saldo": 500,  # saldo inicial generoso para testes
            "limites_usados": {},
            "data_criacao": datetime.now().isoformat(),
            "ultima_recarga_elite": None,
        }
        _historico[usuario_id] = [{
            "tipo": "bonus_boas_vindas",
            "quantidade": PLANOS["free"]["creditos"],
            "timestamp": datetime.now().isoformat(),
        }]
        _data["usuarios"] = _usuarios
        _data["historico"] = _historico
        _save_data(_data)
    return _usuarios[usuario_id]

def get_saldo(usuario_id: str) -> int:
    user = get_usuario(usuario_id)
    if user["plano"] == "elite_lifetime":
        ultima = user.get("ultima_recarga_elite")
        if not ultima or (datetime.now() - datetime.fromisoformat(ultima)) > timedelta(days=30):
            recarga = PLANOS["elite_lifetime"]["recarga_mensal"]
            user["saldo"] += recarga
            user["ultima_recarga_elite"] = datetime.now().isoformat()
            _historico[usuario_id].append({
                "tipo": "recarga_elite_mensal", "quantidade": recarga,
                "timestamp": datetime.now().isoformat(),
            })
            _data["usuarios"] = _usuarios
            _data["historico"] = _historico
            _save_data(_data)
            print(f"[ELITE] {usuario_id} recebeu {recarga} créditos")
    return user["saldo"]

def verificar_saldo(usuario_id: str, quantidade: int) -> int:
    return get_saldo(usuario_id)

def verificar_limite_diario(usuario_id: str, operacao: str) -> bool:
    user = get_usuario(usuario_id)
    plano_config = PLANOS.get(user["plano"], PLANOS["free"])
    limite = plano_config["limites_diarios"].get(operacao, 0)
    if limite == -1: return True
    hoje = datetime.now().date().isoformat()
    if user.get("ultimo_reset_limites") != hoje:
        user["limites_usados"] = {}
        user["ultimo_reset_limites"] = hoje
    return user["limites_usados"].get(operacao, 0) < limite

def incrementar_limite_diario(usuario_id: str, operacao: str):
    user = get_usuario(usuario_id)
    hoje = datetime.now().date().isoformat()
    if user.get("ultimo_reset_limites") != hoje:
        user["limites_usados"] = {}
        user["ultimo_reset_limites"] = hoje
    user["limites_usados"][operacao] = user["limites_usados"].get(operacao, 0) + 1

def checar_rate_limit(usuario_id: str, operacao: str, max_por_minuto: int = 10) -> bool:
    agora = datetime.now()
    if usuario_id not in _rate_limits: _rate_limits[usuario_id] = {}
    if operacao not in _rate_limits[usuario_id]: _rate_limits[usuario_id][operacao] = []
    _rate_limits[usuario_id][operacao] = [
        t for t in _rate_limits[usuario_id][operacao] if agora - t < timedelta(minutes=1)
    ]
    if len(_rate_limits[usuario_id][operacao]) >= max_por_minuto: return False
    _rate_limits[usuario_id][operacao].append(agora)
    return True

def debitar_creditos(usuario_id: str, quantidade: int, operacao: str) -> int:
    user = get_usuario(usuario_id)
    saldo_anterior = user["saldo"]
    novo_saldo = saldo_anterior - quantidade
    if novo_saldo < 0:
        raise ValueError(f"Saldo insuficiente. Tem {saldo_anterior}, precisa de {quantidade}.")
    user["saldo"] = novo_saldo
    if usuario_id not in _historico: _historico[usuario_id] = []
    _historico[usuario_id].append({
        "tipo":"debito","operacao":operacao,"quantidade":quantidade,
        "saldo_anterior":saldo_anterior,"saldo_novo":novo_saldo,
        "timestamp":datetime.now().isoformat(),
    })
    _data["usuarios"] = _usuarios
    _data["historico"] = _historico
    _save_data(_data)
    print(f"[CRÉDITOS] {usuario_id} — {operacao} (-{quantidade}) → saldo: {novo_saldo}")
    return novo_saldo

def creditar(usuario_id: str, quantidade: int, motivo: str = "compra", plano_id: str = None) -> int:
    user = get_usuario(usuario_id)
    saldo_anterior = user["saldo"]
    novo_saldo = saldo_anterior + quantidade
    user["saldo"] = novo_saldo
    if plano_id and plano_id in PLANOS: user["plano"] = plano_id
    if usuario_id not in _historico: _historico[usuario_id] = []
    _historico[usuario_id].append({
        "tipo":"credito","operacao":motivo,"quantidade":quantidade,
        "saldo_anterior":saldo_anterior,"saldo_novo":novo_saldo,
        "plano":plano_id,"timestamp":datetime.now().isoformat(),
    })
    _data["usuarios"] = _usuarios
    _data["historico"] = _historico
    _save_data(_data)
    print(f"[CRÉDITOS] {usuario_id} — {motivo} (+{quantidade}) → saldo: {novo_saldo}")
    return novo_saldo

def historico_creditos(usuario_id: str, limite: int = 50) -> list:
    if usuario_id not in _historico: return []
    return _historico[usuario_id][-limite:]

def calcular_custo(operacao: str, **kwargs) -> int:
    if operacao not in PRECOS: return 1
    base = PRECOS[operacao]
    if operacao == "gerar_video": return base * kwargs.get("duracao", 5)
    if operacao == "gerar_voz": return max(1, (kwargs.get("chars",1000) // 1000) * base)
    return base

def get_plano_info(plano_id: str) -> dict:
    if plano_id not in PLANOS: return None
    return PLANOS[plano_id].copy()