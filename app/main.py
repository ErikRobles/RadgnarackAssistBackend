import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.chat import router as chat_router
from app.api.v1.telegram import router as telegram_router
from app.api.v1.escalation import router as escalation_router

app = FastAPI(title="RadgnarackAssist Backend")

# CORS configuration - local + production domains
allow_origins = [
    # Local development
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

# Add production frontend domain if set
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    allow_origins.append(frontend_url)

# Add API domain if different (for potential future use)
api_domain = os.getenv("API_DOMAIN")
if api_domain and api_domain not in allow_origins:
    allow_origins.append(api_domain)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(telegram_router)
app.include_router(escalation_router)


@app.get("/")
def root():
    return {"status": "ok", "service": "RadgnarackAssist Backend"}