# main.py
"""
ARES ALPHA Dashboard Server
============================
메인 실행 진입점

Usage:
    python main.py
    
    # 또는 직접 uvicorn 사용
    uvicorn monitoring.api:app --host 0.0.0.0 --port 8000 --reload
"""

import uvicorn

if __name__ == "__main__":
    # ARES Dashboard Server 실행
    # host="0.0.0.0"으로 설정하여 외부 접속 허용
    print(">>> Starting ARES ALPHA Dashboard (v2.4 Backend)...")
    print(">>> Access: http://localhost:8000")
    uvicorn.run("monitoring.api:app", host="0.0.0.0", port=8000, reload=True)
