from fastapi import FastAPI
from .info import info_router
from .session import session_router, cursor, connection
from .settings import settings_router
from .chat import chat_router


cursor.executescript(
    """
    PRAGMA foreign_keys = ON;
    CREATE TABLE IF NOT EXISTS user_session ( uuid varchar(36) PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL ,session_name varchar(250) UNIQUE );
    CREATE TABLE IF NOT EXISTS chat_history ( uuid varchar(36) REFERENCES user_session(uuid) ON DELETE CASCADE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL, chat BOLB , role VARCHAR(25) DEFAULT "user" NOT NULL);
    CREATE TABLE IF NOT EXISTS file_records ( uuid varchar(36) REFERENCES user_session(uuid) ON DELETE CASCADE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL, file_name varchar(250) NOT NULL, file_id varchar(64) NOT NULL);
    """
)

connection.commit()


app = FastAPI()

app.include_router(info_router)
app.include_router(session_router)
app.include_router(settings_router)
app.include_router(chat_router)
