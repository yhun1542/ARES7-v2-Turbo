# monitoring/api.py
"""
ARES Alpha - Dashboard API (v2.4)
==================================
FastAPI 기반 REST API + 대시보드 서빙

엔드포인트:
- GET  /        : 대시보드 HTML 페이지
- GET  /status  : 현재 시스템 상태 (JSON)
- POST /kill_switch : Kill Switch 제어
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

from monitoring.state import KillSwitchMode
from monitoring.store import load_state, set_kill_switch_mode

app = FastAPI(title="ARES ALPHA Dashboard Backend")

# CORS 설정 (개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# 1. 정적 파일 (Fonts, CSS 등)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 2. HTML 템플릿 설정
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
else:
    templates = None


# 요청 모델
class KillSwitchRequest(BaseModel):
    mode: KillSwitchMode


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """대시보드 메인 페이지 렌더링"""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        # Fallback: 간단한 HTML
        return HTMLResponse(content="""
        <html>
        <head><title>ARES ALPHA</title></head>
        <body style="background:#0b0c10; color:#fff; font-family:sans-serif; padding:40px;">
            <h1>ARES ALPHA Dashboard</h1>
            <p>templates/index.html 파일이 없습니다.</p>
            <p><a href="/status" style="color:#00bcd4;">View Status JSON</a></p>
        </body>
        </html>
        """)


@app.get("/status")
async def get_status():
    """프론트엔드에서 주기적으로 호출하는 상태 API"""
    state = load_state()
    # 실제 운영시에는 여기서 live_orchestrator로부터 최신 값을 갱신받을 수 있음
    return state.to_dict()


@app.post("/kill_switch")
async def change_kill_switch(req: KillSwitchRequest):
    """Kill Switch 제어 API"""
    print(f"!!! [SYSTEM ALERT] KILL SWITCH ACTIVATED: {req.mode} !!!")
    new_state = set_kill_switch_mode(req.mode)
    return {"status": "ok", "current_mode": new_state.kill_switch}


@app.get("/health")
async def health_check():
    """서버 헬스 체크"""
    return {"status": "healthy", "service": "ares-alpha-dashboard"}


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """서버 실행"""
    import uvicorn
    print(">>> Starting ARES ALPHA Dashboard (v2.4 Backend)...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
