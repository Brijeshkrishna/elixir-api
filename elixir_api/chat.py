from pathlib import Path
import subprocess
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from .session import get_session_id, is_valid_session_id, cursor, connection, return_id
import uuid
import requests
import json
from pydantic import BaseModel
import hashlib
import qdrant_client
from typing import Union, Literal
import re

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .history import get_history
from deep_translator import GoogleTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter

QDRANT_URL = "http://localhost:6333"  # ":memory:"
OLLAMA_URL = "http://localhost:11434"

MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"

QRANT_CLIENT = qdrant_client.QdrantClient(location=QDRANT_URL)

DOCUMENT_SOURCE_DIR = "documents/"
TMP_DIR = "tmp/"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

Path(DOCUMENT_SOURCE_DIR).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
subprocess.call(["ollama", "pull", MODEL, EMBEDDING_MODEL])

CHATMODEL = ChatOllama(
    base_url=OLLAMA_URL, model="mistral", temperature=0, verbose=True
)

embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)


chat_router = APIRouter()


def inserted_db(
    session_id: uuid.UUID, role: Literal["user", "assistant"], message: str
):
    cursor.execute(
        "INSERT INTO chat_history(uuid,role,chat) VALUES(?,?,?)",
        (session_id, role, message),
    )


async def chat_responder(message, session_id):

    message = text_process(message=message)
    inserted_db(session_id=session_id, role="user", message=message)

    histroy = []
    for msg in (await get_history(session_id))["history"]:
        if msg["role"] == "user":
            histroy.append(HumanMessage(content=msg["message"]))
        else:
            histroy.append(AIMessage(content=msg["message"]))

    gen_message = ""
    async for resp_stream in CHATMODEL.astream(histroy):
        gen_message = gen_message + resp_stream.content
        yield resp_stream.content

    inserted_db(session_id=session_id, role="assistant", message=gen_message)
    connection.commit()


@chat_router.get("/chat")
async def chat(session_id_or_name: Union[str, uuid.UUID], message: str):
    session_id_or_name = await return_id(session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}

    return StreamingResponse(
        chat_responder(message, session_id_or_name), media_type="text/event-stream"
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

    document = PyPDFLoader(pdf_file_name, extract_images=False).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(document)

    for idx, val in enumerate(texts):
        val = re.sub(r"\\u[0-9a-b]{0,4}", " ", val.page_content)
        texts[idx].page_content = GoogleTranslator(
            source="auto", target="en"
        ).translate(val.strip())

    Qdrant.from_documents(
        location=QDRANT_URL,
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


@chat_router.get("/delete_file")
async def delete_file(session_id_or_name, file_id):
    session_id_or_name = await return_id(session_id_or_name)
    if await return_id(session_id_or_name) == None:
        return {"status": "bad", "message": "session not exits"}

    cursor.execute(
        "DELETE FROM file_records WHERE uuid = ? AND file_id = ?",
        (session_id_or_name, file_id),
    )
    connection.commit()
    return {"status": "Done"}


@chat_router.get("/delete_all_file")
async def delete_file(session_id_or_name):
    session_id_or_name = await return_id(session_id_or_name)
    if await return_id(session_id_or_name) == None:
        return {"status": "bad", "message": "session not exits"}

    cursor.execute(
        "DELETE FROM file_records WHERE uuid = ?",
        (session_id_or_name, file_id),
    )
    connection.commit()
    return {"status": "Done"}


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

    resp = model.invoke(histroy)

    result = resp["result"]

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
    print(result)
    return result


@chat_router.get("/chat_with_file")
async def file_search(session_id_or_name: str, message: str, file_id: str):
    session_id_or_name = await return_id(session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}

    collection_list = cursor.execute(
        "SELECT file_id FROM file_records fr JOIN user_session us ON fr.uuid = us.uuid AND fr.file_id = ?",
        (file_id,),
    ).fetchall()

    if collection_list is None or collection_list == [] or collection_list == ():
        return {"status": "error", "msg": "file_id not found"}
        # return await chat_responder_file(
        #     message=message, session_id=session_id_or_name, model=CHATMODEL
        # )

    # collection_list = [i[0] for i in collection_list]

    qd = Qdrant(
        client=QRANT_CLIENT,
        embeddings=embedding_function,
        collection_name=file_id,
    )

    r = RetrievalQA.from_chain_type(
        llm=CHATMODEL,
        chain_type="stuff",
        retriever=qd.as_retriever(),
        return_source_documents=True,
    )

    return await chat_responder_file(message, session_id_or_name, r)


def text_process(message: str) -> str:
    return GoogleTranslator(source="auto", target="en").translate(message.strip())
