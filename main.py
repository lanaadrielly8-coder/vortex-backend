from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import asyncio
import os
import uvicorn

try:
    from dotenv import load_dotenv
    load_dotenv(".env/.env")
except ImportError:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Atlas Cloud ──────────────────────────────────────────────
ATLAS_API_KEY  = os.getenv("ATLAS_API_KEY", "")
ATLAS_BASE_URL = "https://api.atlascloud.ai/api/v1/model"

# ── Mercado Pago ─────────────────────────────────────────────
MP_ACCESS_TOKEN = os.getenv("MP_ACCESS_TOKEN", "")
VORTEX_URL      = os.getenv("VORTEX_URL", "http://localhost:5173")

MODELOS = {
    "kling-t2v":    "kling-v3/text-to-video",
    "kling-i2v":    "kling-v3/image-to-video",
    "seedance-t2v": "bytedance/seedance-2.0/text-to-video",
    "seedance-i2v": "bytedance/seedance-2.0/image-to-video",
    "veo-t2v":      "google/veo-3.1/text-to-video",
}


# ── Schemas ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    texto: str

class ImageRequest(BaseModel):
    prompt: str

class VideoRequest(BaseModel):
    prompt: str
    modelo: Optional[str] = "kling-t2v"
    image_url: Optional[str] = None
    duracao: Optional[int] = 5
    resolucao: Optional[str] = "720p"
    ratio: Optional[str] = "16:9"
    audio: Optional[bool] = False
    negative_prompt: Optional[str] = ""

class PagamentoRequest(BaseModel):
    pacote_id: str
    preco: float
    creditos: int
    descricao: str


# ── Rotas existentes ─────────────────────────────────────────

@app.get("/status")
async def get_status():
    return {
        "modelo": "Mistral",
        "status": "online",
        "atlas_cloud": "configurado" if ATLAS_API_KEY else "ATLAS_API_KEY nao definida",
        "mercado_pago": "configurado" if MP_ACCESS_TOKEN else "MP_ACCESS_TOKEN nao definido",
    }

@app.get("/perfil")
async def get_perfil():
    return {"canal": "Modo: Criativo", "plataforma": "TikTok e YouTube"}

@app.get("/historico")
async def get_historico():
    return []

@app.post("/chat")
async def chat(request: ChatRequest):
    return {"resposta": f"Vortex recebeu sua mensagem: {request.texto}"}

@app.post("/gerar-imagem")
async def gerar_imagem(request: ImageRequest):
    try:
        image_url = "https://picsum.photos/1024/1024"
        return {"ok": True, "imagem": image_url}
    except Exception as e:
        return {"ok": False, "erro": str(e)}


# ── Gerar vídeo ──────────────────────────────────────────────

@app.post("/gerar-video")
async def gerar_video(request: VideoRequest):
    if not ATLAS_API_KEY:
        raise HTTPException(status_code=500, detail="ATLAS_API_KEY nao configurada.")

    model_id = MODELOS.get(request.modelo)
    if not model_id:
        raise HTTPException(status_code=400, detail=f"Modelo invalido. Opcoes: {list(MODELOS.keys())}")

    if "i2v" in request.modelo and not request.image_url:
        raise HTTPException(status_code=400, detail="image_url obrigatorio para image-to-video")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ATLAS_API_KEY}",
    }

    payload = {
        "model": model_id,
        "prompt": request.prompt,
        "duration": request.duracao,
        "resolution": request.resolucao,
        "ratio": request.ratio,
        "watermark": False,
    }

    if request.negative_prompt:
        payload["negative_prompt"] = request.negative_prompt
    if request.audio:
        payload["generate_audio"] = True
    if request.image_url:
        payload["image_url"] = request.image_url

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                f"{ATLAS_BASE_URL}/generateVideo",
                headers=headers,
                json=payload,
            )
            data = res.json()
            print("ATLAS RESPOSTA:", data)

        if res.status_code != 200 or not data.get("data", {}).get("id"):
            raise HTTPException(status_code=502, detail=f"Atlas Cloud erro: {data}")

        prediction_id = data["data"]["id"]

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Erro de conexao: {str(e)}")

    poll_url = f"{ATLAS_BASE_URL}/prediction/{prediction_id}"
    timeout  = 300
    interval = 4
    elapsed  = 0

    async with httpx.AsyncClient(timeout=15) as client:
        while elapsed < timeout:
            await asyncio.sleep(interval)
            elapsed += interval

            try:
                poll_res  = await client.get(poll_url, headers=headers)
                poll_data = poll_res.json()
                print("POLL STATUS:", poll_data.get("data", {}).get("status"))
            except Exception:
                continue

            status = poll_data.get("data", {}).get("status", "")

            if status == "succeeded":
                videos = poll_data["data"].get("output", [])
                if not videos:
                    raise HTTPException(status_code=502, detail="Video gerado mas sem URL.")
                video_url = videos[0] if isinstance(videos, list) else videos
                return {
                    "ok": True,
                    "video_url": video_url,
                    "prediction_id": prediction_id,
                    "modelo_usado": model_id,
                    "duracao": request.duracao,
                }

            if status == "failed":
                erro = poll_data.get("data", {}).get("error", "sem detalhe")
                raise HTTPException(status_code=502, detail=f"Geracao falhou: {erro}")

    raise HTTPException(status_code=504, detail="Timeout: demorou mais que 5 minutos.")


@app.get("/modelos-video")
async def listar_modelos():
    return {
        "modelos": [
            {"id": "kling-t2v",    "nome": "Kling 3.0 - Texto para Video",    "preco_por_seg": 0.153},
            {"id": "kling-i2v",    "nome": "Kling 3.0 - Imagem para Video",   "preco_por_seg": 0.153},
            {"id": "seedance-t2v", "nome": "Seedance 2.0 - Texto para Video", "preco_por_seg": 0.10},
            {"id": "seedance-i2v", "nome": "Seedance 2.0 - Imagem para Video","preco_por_seg": 0.10},
            {"id": "veo-t2v",      "nome": "Veo 3.1 - Texto para Video",      "preco_por_seg": 0.20},
        ]
    }


# ── Mercado Pago ─────────────────────────────────────────────

@app.post("/pagamento/criar")
async def criar_pagamento(request: PagamentoRequest):
    if not MP_ACCESS_TOKEN:
        raise HTTPException(500, "MP_ACCESS_TOKEN nao configurado no .env")

    try:
        import mercadopago
        mp = mercadopago.SDK(MP_ACCESS_TOKEN)
    except ImportError:
        raise HTTPException(500, "Instale: pip install mercadopago")

    preference_data = {
        "items": [
            {
                "title": request.descricao,
                "quantity": 1,
                "unit_price": float(request.preco),
                "currency_id": "BRL",
            }
        ],
        "back_urls": {
            "success": f"{VORTEX_URL}/creditos/sucesso?pacote={request.pacote_id}&creditos={request.creditos}",
            "failure": f"{VORTEX_URL}/creditos?erro=pagamento_recusado",
            "pending": f"{VORTEX_URL}/creditos?status=pendente",
        },
        "auto_return": "approved",
        "notification_url": "http://127.0.0.1:8081/pagamento/webhook",
        "metadata": {
            "pacote_id": request.pacote_id,
            "creditos": request.creditos,
        },
        "payment_methods": {
            "installments": 12,
        },
    }

    result = mp.preference().create(preference_data)

    if result["status"] != 201:
        raise HTTPException(502, f"Mercado Pago erro: {result}")

    preference = result["response"]
    checkout_url = preference.get("sandbox_init_point") or preference.get("init_point")

    return {
        "checkout_url": checkout_url,
        "preference_id": preference["id"],
    }


@app.post("/pagamento/webhook")
async def webhook_pagamento(request: Request):
    body = await request.json()
    print("WEBHOOK MP:", body)

    tipo = body.get("type")
    if tipo == "payment":
        payment_id = body["data"]["id"]

        try:
            import mercadopago
            mp = mercadopago.SDK(MP_ACCESS_TOKEN)
            payment = mp.payment().get(payment_id)

            if payment["status"] == 200:
                p = payment["response"]
                if p.get("status") == "approved":
                    meta     = p.get("metadata", {})
                    pacote   = meta.get("pacote_id")
                    creditos = meta.get("creditos", 0)
                    print(f"Pagamento aprovado! Pacote: {pacote}, Creditos: {creditos}")
                    # TODO: adicionar creditos na conta do usuario no banco de dados
        except Exception as e:
            print("Erro no webhook:", e)

    return {"ok": True}


# ── Iniciar servidor ─────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)
