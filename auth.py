import os, secrets
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, Request
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
VORTEX_URL = os.getenv("VORTEX_URL","http://127.0.0.1:8082")
FRONTEND_URL = os.getenv("FRONTEND_URL","http://localhost:5173")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID","")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET","")
def criar_token(uid,email,nome):
    return jwt.encode({"sub":uid,"email":email,"nome":nome,"exp":datetime.utcnow()+timedelta(days=30)},JWT_SECRET,algorithm="HS256")
def verificar_token(token):
    try: return jwt.decode(token,JWT_SECRET,algorithms=["HS256"])
    except: raise HTTPException(401,"Token invalido")
def get_usuario_token(request):
    token=request.cookies.get("vortex_token") or request.headers.get("Authorization","").replace("Bearer ","")
    if not token: return None
    try: return verificar_token(token)
    except: return None
def google_auth_url():
    if not GOOGLE_CLIENT_ID: return ""
    from urllib.parse import urlencode
    return "https://accounts.google.com/o/oauth2/v2/auth?"+urlencode({"client_id":GOOGLE_CLIENT_ID,"redirect_uri":f"{VORTEX_URL}/auth/google/callback","response_type":"code","scope":"openid email profile"})
async def google_callback(code):
    import httpx
    if not GOOGLE_CLIENT_ID: raise HTTPException(500,"Google OAuth nao configurado")
    async with httpx.AsyncClient() as c:
        r=await c.post("https://oauth2.googleapis.com/token",data={"code":code,"client_id":GOOGLE_CLIENT_ID,"client_secret":GOOGLE_CLIENT_SECRET,"redirect_uri":f"{VORTEX_URL}/auth/google/callback","grant_type":"authorization_code"})
        r2=await c.get("https://www.googleapis.com/oauth2/v3/userinfo",headers={"Authorization":f"Bearer {r.json()['access_token']}"})
        return r2.json()
