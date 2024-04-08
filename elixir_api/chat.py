from pathlib import Path
import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from .session import get_session_id, is_valid_session_id, cursor, connection, return_id
import uuid
import requests
import json
from pydantic import BaseModel
import hashlib
import qdrant_client

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage


QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"

MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"

QRANT_CLIENT = qdrant_client.QdrantClient(url=QDRANT_URL)

DOCUMENT_SOURCE_DIR = "documents/"
TMP_DIR = "tmp/"


Path(DOCUMENT_SOURCE_DIR).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
os.system(f"ollama pull {MODEL} {EMBEDDING_MODEL}")

model = ChatOllama(
    base_url=OLLAMA_URL, model="mistral", temperature=1, verbose=True, format="json"
)

embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)

# embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class Chat(BaseModel):
    message: str
    session_id_or_name: str


chat_router = APIRouter()


@chat_router.get("/get_chat_history")
async def get_chat_history_of(session_id_or_name: str):

    session_id_or_name = await return_id(session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}
    return await get_history(session_id_or_name)


async def get_history(session_id: str):
    res = cursor.execute(
        "SELECT ch.role, ch.chat FROM user_session us JOIN chat_history ch ON  ch.uuid = ? AND ch.uuid = us.uuid order by ch.created_at",
        (session_id,),
    )
    res = res.fetchall()

    return {"history": [{"role": i[0], "message": i[1]} for i in res]}


async def chat_responder(message, session_id, model):

    cursor.execute(
        "INSERT INTO chat_history(uuid,role,chat) VALUES(?,?,?)",
        (session_id, "user", message),
    )
    connection.commit()

    histroy = []
    for msg in (await get_history(session_id))["history"]:
        if msg["role"] == "user":
            histroy.append(HumanMessage(content=msg["message"]))
        else:
            histroy.append(AIMessage(content=msg["message"]))
    print(histroy)

    gen_message = ""
    for resp_stream in model.stream(histroy):
        print(resp_stream)
        gen_message = gen_message + resp_stream.content
        yield resp_stream.content

    cursor.execute(
        "INSERT INTO chat_history(uuid,role,chat) VALUES(?,?,?)",
        (session_id, "assistant", gen_message),
    )
    connection.commit()


@chat_router.post("/chat")
async def read_item(item: Chat):
    session_id_or_name = await return_id(item.session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}

    return StreamingResponse(
        chat_responder(item.message, session_id_or_name, model),
        media_type="text/event-stream",
    )


class FileUploadBase(BaseModel):
    file: UploadFile
    session_id_or_name: str


@chat_router.post("/upload_file")
async def read_item(session_id_or_name: str, item: UploadFile = File(...)):
    session_id_or_name = await return_id(session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}

    if item.content_type != "application/pdf":
        return {"status": "bad", "message": "accpect only pdf"}

    with open(DOCUMENT_SOURCE_DIR + item.filename, "wb") as f:
        f.write(await item.read())

    file_id = await load_file(DOCUMENT_SOURCE_DIR + item.filename)
    cursor.execute(
        "INSERT INTO file_records(uuid,file_name,file_id) VALUES(?,?,?)",
        (session_id_or_name, item.filename, file_id),
    )
    connection.commit()
    return {"status": "ok", "message": "file inserted"}


async def load_file(pdf_file_name):

    with open(pdf_file_name, "rb") as f:
        file_contents = f.read()

    collection_name = hashlib.sha256(file_contents).hexdigest()

    if check_collection_exists(collection_name):
        return collection_name

    document = PyPDFLoader(pdf_file_name, extract_images=False).load_and_split()

    # db = Qdrant(
    #     client=QRANT_CLIENT,
    #     collection_name=collection_name,
    #     embeddings=embedding_function,
    # )
    # print(db)
    # db.add_documents(document)
    Qdrant.from_documents(
        url=QDRANT_URL,
        documents=document,
        embedding=embedding_function,
        collection_name=collection_name,
    )

    return collection_name


def get_collection():
    return [i.name for i in QRANT_CLIENT.get_collections().collections]


def check_collection_exists(collection_name):
    return collection_name in get_collection()


@chat_router.get("/list_files")
async def read_item(session_id_or_name):
    if await return_id(session_id_or_name) == None:
        return {"status": "bad", "message": "session not exits"}

    collection_list = cursor.execute(
        "SELECT fr.file_id,fr.file_name FROM  file_records fr JOIN user_session us ON fr.uuid = us.uuid"
    ).fetchall()

    return collection_list


@chat_router.get("/file_search")
async def file_search(file_id: str, search: str):
    qd = Qdrant(
        client=QRANT_CLIENT, embeddings=embedding_function, collection_name=file_id
    )
    return qd.similarity_search_with_score(search)


def get_relavent_collection(message, collection_name):
    scores = {}

    for i in collection_name:
        scores[i] = max(
            (get_vector_store(i)).similarity_search_with_score(message),
            key=lambda x: x[1],
        )[1]

    return max(scores, key=scores.get)


def get_vector_store(collection_name):
    return Qdrant(
        client=QRANT_CLIENT,
        collection_name=collection_name,
        embeddings=embedding_function,
    )


async def chat_responder_file(message, session_id, model):
    print(000)
    cursor.execute(
        "INSERT INTO chat_history(uuid,role,chat) VALUES(?,?,?)",
        (session_id, "user", message),
    )
    connection.commit()

    histroy = []
    for msg in (await get_history(session_id))["history"]:
        if msg["role"] == "user":
            histroy.append(HumanMessage(content=msg["message"]))
        else:
            histroy.append(AIMessage(content=msg["message"]))
    print(histroy)

    # gen_message = ""
    # for resp_stream in model.stream(message):
    #     gen_message = gen_message + resp_stream['result']
    #     # print(resp_stream,end='')
    #     yield resp_stream['result']

    resp = model.invoke(histroy)

    result = resp["result"][1:]
    print(result)

    source = {}
    for i in resp["source_documents"]:
        source[i.metadata["page"]] = i.metadata["source"].split("/")[-1]

    result = result + "\n"
    for i in source.keys():
        result = format(f"{result}page {i} {source[i]}\n")

    cursor.execute(
        "INSERT INTO chat_history(uuid,role,chat) VALUES(?,?,?)",
        (session_id, "assistant", result),
    )
    connection.commit()
    return result


@chat_router.get("/chat_with_file")
async def file_search(session_id_or_name: str, message: str):
    session_id_or_name = await return_id(session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}

    collection_list = cursor.execute(
        "SELECT file_id FROM file_records fr JOIN user_session us ON fr.uuid = us.uuid"
    ).fetchall()

    if collection_list is None or collection_list == [] or collection_list == ():

        return await chat_responder_file(
            message=message, session_id=session_id_or_name, model=model
        )

    collection_list = [i[0] for i in collection_list]

    qd = Qdrant(
        client=QRANT_CLIENT,
        embeddings=embedding_function,
        collection_name=get_relavent_collection(message, collection_list),
    )

    r = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=qd.as_retriever(),
        return_source_documents=True,
    )

    return await chat_responder_file(
        message=message, session_id=session_id_or_name, model=r
    )
