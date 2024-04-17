import threading
import base64
import hashlib
import re
import configparser
import concurrent.futures
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.ollama import OllamaEmbeddings

from fastapi import APIRouter, File, UploadFile
import qdrant_client

from .session import return_id
from .init import __embedding_function__, __qdrant_clinet__, __connection__
from .helper import FileUploadBase, get_collection, get_hash,text_process,commit,check_collection_exists


__CONFIG__ = configparser.ConfigParser()
__CONFIG__.read("elixir.ini")

file_handel_router = APIRouter()

cursor = __connection__.cursor()


@file_handel_router.post("/upload_file")
async def read_item(item: FileUploadBase):
    session_id_or_name = await return_id(item.session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}

    file_ext = item.filename.split(".")[-1]
    if file_ext != "pdf" and file_ext != "txt":
        return {"status": "bad", "message": "accpect only pdf"}

    file_name = __CONFIG__.get("FOLDERS", "DOCUMENT_SOURCE_DIR") + item.filename

    with open(file_name, "wb") as f:
        f.write(bytes(json.loads(item.file)))

    file_id = await load_file(file_name)
    cursor.execute(
        "INSERT INTO file_records(uuid,file_name,file_id) VALUES(?,?,?)",
        (session_id_or_name, item.filename, file_id),
    )
    commit()
    return {"status": "ok", "message": "file inserted"}


# def generate_image_describtion(image:bytes):
#     return IMAGE_MODE.bind(images=[image.decode("UTF-8")]).invoke("describe the image")


async def load_file(pdf_file_name):

    collection_name = get_hash(pdf_file_name)

    if check_collection_exists(collection_name):
        return collection_name

    file_ext = pdf_file_name.split(".")[-1]

    if file_ext == "pdf":
        texts = PyPDFLoader(pdf_file_name, extract_images=False).load_and_split()
    else:
        texts = TextLoader(pdf_file_name).load_and_split()
        
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=int(__CONFIG__.get("FILE_CONFIG", "CHUNK_SIZE")),
        #     chunk_overlap=int(__CONFIG__.get("FILE_CONFIG", "CHUNK_OVERLAP")),
        # )
        # texts = text_splitter.split_documents(document)

    for idx, val in enumerate(texts):
        val = re.sub(r"\\u[0-9a-b]{0,4}", " ", val.page_content)
        texts[idx].page_content = text_process(val.strip())

    Qdrant.from_documents(
        url=__CONFIG__.get("STORAGE", "QDRANT_URL"),
        documents=texts,
        embedding=__embedding_function__,
        collection_name=collection_name,
    )

    return collection_name


@file_handel_router.get("/list_files")
async def list_files(session_id_or_name):
    if await return_id(session_id_or_name) == None:
        return {"status": "bad", "message": "session not exits"}

    collection_list = cursor.execute(
        "SELECT fr.file_id,fr.file_name FROM  file_records fr JOIN user_session us ON fr.uuid = us.uuid"
    ).fetchall()

    return collection_list


# @file_handel_router.get("/delete_file")
# async def delete_file(session_id_or_name, file_id):
#     session_id_or_name = await return_id(session_id_or_name)
#     if await return_id(session_id_or_name) == None:
#         return {"status": "bad", "message": "session not exits"}

#     cursor.execute(
#         "DELETE FROM file_records WHERE uuid = ? AND file_id = ?",
#         (session_id_or_name, file_id),
#     )
#     connection.commit()
#     return {"status": "Done"}


# @file_handel_router.get("/delete_all_file")
# async def delete_file(session_id_or_name):
#     session_id_or_name = await return_id(session_id_or_name)
#     if await return_id(session_id_or_name) == None:
#         return {"status": "bad", "message": "session not exits"}

#     cursor.execute(
#         "DELETE FROM file_records WHERE uuid = ?",
#         (session_id_or_name,),
#     )
#     connection.commit()
#     return {"status": "Done"}
