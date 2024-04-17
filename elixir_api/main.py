from fastapi import FastAPI
from .session import session_router
from .settings import settings_router
from .chat import chat_router
from .history import history_router
from .file_handel import file_handel_router
import uvicorn

app = FastAPI(redoc_url="/redoc")

app.include_router(session_router)
app.include_router(settings_router)
app.include_router(chat_router)
app.include_router(history_router)
app.include_router(file_handel_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,)