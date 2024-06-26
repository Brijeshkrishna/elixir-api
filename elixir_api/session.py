from fastapi import APIRouter
import subprocess
import sqlite3
from typing import Union
import uuid
from .init import __connection__

session_router = APIRouter()

cursor = __connection__.cursor()


@session_router.get("/new_session")
async def new_session(session_name: str):
    session_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO user_session(`uuid`,`session_name`) VALUES(?,?)",
        (session_id, session_name),
    )
    __connection__.commit()
    return {"status": "ok", "session_id": session_id, "session_name": session_name}


@session_router.get("/list_sessions")
async def list_sessions():
    res = cursor.execute("SELECT * from user_session")
    session_list = []
    for i in res.fetchall():
        responses = {}
        responses["session_id"] = i[0]
        responses["created_at"] = i[1]
        responses["session_name"] = i[2]
        session_list.append(responses)
    return session_list


@session_router.get("/delete_session_by_id")
async def delete_session_by_id(session_id: Union[uuid.UUID, str]):
    session_id = await return_id(session_id)
    if session_id == None:
        return {"error": "no uuid found"}

    cursor.execute("DELETE FROM user_session where uuid = ?", (session_id,))
    __connection__.commit()
    return {"status": "ok"}


async def get_session_id(session_name) -> uuid.UUID | None:
    res = cursor.execute(
        "SELECT uuid from user_session where session_name = ? ", (session_name,)
    )
    res = res.fetchone()
    if res is None or res == [] or res == ():
        return None
    return res[0]


async def is_valid_session_id(session_id) -> bool:
    res = cursor.execute("SELECT uuid from user_session where uuid = ? ", (session_id,))
    res = res.fetchall()
    if res is None or res == [] or res == ():
        return False
    return True


async def return_id(session_id_or_name) -> uuid.UUID:
    if await is_valid_session_id(session_id_or_name) == False:
        return await get_session_id(session_id_or_name)
    return session_id_or_name
