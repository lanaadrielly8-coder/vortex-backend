"""
VORTEX — Banco de dados com Supabase
Substitui o JSON local por banco real na nuvem
"""
import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

# Tenta usar Supabase, fallback para JSON local
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

_usar_supabase = bool(SUPABASE_URL and SUPABASE_KEY)

if _usar_supabase:
    try:
        from supabase import create_client, Client
        _supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[DB] ✅ Supabase conectado")
    except ImportError:
        print("[DB] ⚠️ supabase não instalado — usando JSON local")
        _usar_supabase = False
else:
    print("[DB] ℹ️ Sem Supabase configurado — usando JSON local")

# ── JSON local (fallback) ──────────────────────────────────────────────────────
DB_FILE = "vortex_db.json"

def _load_db() -> dict:
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"usuarios": {}, "roteiros": [], "geracoes": []}

def _save_db(data: dict):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── USUÁRIOS ──────────────────────────────────────────────────────────────────
def get_usuario_db(usuario_id: str) -> dict:
    if _usar_supabase:
        try:
            r = _supabase.table("usuarios").select("*").eq("id", usuario_id).execute()
            if r.data:
                return r.data[0]
        except Exception as e:
            print(f"[DB] erro get_usuario: {e}")
    # fallback JSON
    db = _load_db()
    return db["usuarios"].get(usuario_id, {
        "id": usuario_id,
        "plano": "free",
        "creditos": 50,
        "email": "",
        "nome": "",
        "nicho": "",
        "criado_em": datetime.now().isoformat(),
    })

def salvar_usuario_db(usuario_id: str, dados: dict):
    if _usar_supabase:
        try:
            dados["id"] = usuario_id
            dados["atualizado_em"] = datetime.now().isoformat()
            _supabase.table("usuarios").upsert(dados).execute()
            return
        except Exception as e:
            print(f"[DB] erro salvar_usuario: {e}")
    # fallback JSON
    db = _load_db()
    if "usuarios" not in db:
        db["usuarios"] = {}
    db["usuarios"][usuario_id] = {**db["usuarios"].get(usuario_id, {}), **dados}
    _save_db(db)

def get_creditos_db(usuario_id: str) -> int:
    u = get_usuario_db(usuario_id)
    return int(u.get("creditos", 50))

def set_creditos_db(usuario_id: str, creditos: int):
    if _usar_supabase:
        try:
            _supabase.table("usuarios").upsert({
                "id": usuario_id,
                "creditos": creditos,
                "atualizado_em": datetime.now().isoformat()
            }).execute()
            return
        except Exception as e:
            print(f"[DB] erro set_creditos: {e}")
    db = _load_db()
    if "usuarios" not in db:
        db["usuarios"] = {}
    if usuario_id not in db["usuarios"]:
        db["usuarios"][usuario_id] = {}
    db["usuarios"][usuario_id]["creditos"] = creditos
    _save_db(db)

# ── HISTÓRICO DE GERAÇÕES ─────────────────────────────────────────────────────
def salvar_geracao_db(usuario_id: str, tipo: str, dados: dict):
    geracao = {
        "usuario_id": usuario_id,
        "tipo": tipo,
        "dados": dados,
        "criado_em": datetime.now().isoformat(),
    }
    if _usar_supabase:
        try:
            _supabase.table("geracoes").insert(geracao).execute()
            return
        except Exception as e:
            print(f"[DB] erro salvar_geracao: {e}")
    # fallback JSON
    db = _load_db()
    if "geracoes" not in db:
        db["geracoes"] = []
    db["geracoes"].append(geracao)
    db["geracoes"] = db["geracoes"][-200:]  # mantém últimas 200
    _save_db(db)

def get_geracoes_db(usuario_id: str, tipo: str = None, limite: int = 50) -> list:
    if _usar_supabase:
        try:
            q = _supabase.table("geracoes").select("*").eq("usuario_id", usuario_id)
            if tipo:
                q = q.eq("tipo", tipo)
            r = q.order("criado_em", desc=True).limit(limite).execute()
            return r.data or []
        except Exception as e:
            print(f"[DB] erro get_geracoes: {e}")
    # fallback JSON
    db = _load_db()
    geracoes = [g for g in db.get("geracoes", []) if g.get("usuario_id") == usuario_id]
    if tipo:
        geracoes = [g for g in geracoes if g.get("tipo") == tipo]
    return geracoes[-limite:][::-1]

# ── PERFIL E MEMÓRIA ─────────────────────────────────────────────────────────
def salvar_perfil_db(usuario_id: str, perfil: dict):
    salvar_usuario_db(usuario_id, {"perfil": perfil, "nicho": perfil.get("nicho","")})

def get_perfil_db(usuario_id: str) -> dict:
    u = get_usuario_db(usuario_id)
    return u.get("perfil", {})

# ── ROTEIROS SALVOS ───────────────────────────────────────────────────────────
def salvar_roteiro_db(usuario_id: str, tema: str, roteiro: str):
    salvar_geracao_db(usuario_id, "roteiro", {"tema": tema, "roteiro": roteiro})

def get_roteiros_db(usuario_id: str, limite: int = 20) -> list:
    return get_geracoes_db(usuario_id, tipo="roteiro", limite=limite)