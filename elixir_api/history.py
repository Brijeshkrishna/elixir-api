from fastapi import APIRouter, File, UploadFile
from typing import Union
from .session import cursor, return_id, connection
import uuid

history_router = APIRouter()


@history_router.get("/get_chat_history")
async def get_chat_history(session_id_or_name: Union[str, uuid.UUID]):

    session_id_or_name = await return_id(session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}
    return await get_history(session_id_or_name)


@history_router.get("/clear_chat_history")
async def get_chat_history(session_id_or_name: Union[str, uuid.UUID]):

    session_id_or_name = await return_id(session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}

    cursor.execute(
        "DELETE FROM chat_history ch WHERE ch.uuid= ?", (session_id_or_name,)
    )
    connection.commit()

    return {"status": "good", "message": "history cleared"}


async def get_history(session_id: str):
    res = cursor.execute(
        "SELECT ch.role, ch.chat FROM user_session us JOIN chat_history ch ON  ch.uuid = ? AND ch.uuid = us.uuid order by ch.created_at",
        (session_id,),
    )
    res = res.fetchall()
    return {"history": [{"role": i[0], "message": i[1]} for i in res]}
