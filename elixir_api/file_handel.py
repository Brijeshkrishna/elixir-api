from fastapi import APIRouter
from .session import get_session_id, is_valid_session_id, cursor, connection, return_id
import hashlib
from fastapi import APIRouter, File, UploadFile
import base64
from langchain_community.llms import Ollama
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
import threading

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
import re
from io import BytesIO
from PIL import Image

EMBEDDING_MODEL = "nomic-embed-text"

TMP_DIR = "images/"

IMAGE_MODE = Ollama(model="llava")
DOCUMENT_SOURCE_DIR = "documents/"
import qdrant_client
from langchain.embeddings.ollama import OllamaEmbeddings

import concurrent.futures


QDRANT_URL = "http://localhost:6333"  # ":memory:"

QRANT_CLIENT = qdrant_client.QdrantClient(url=QDRANT_URL)

from deep_translator import GoogleTranslator

file_handel_router = APIRouter()
embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)


def get_collection():
    return [i.name for i in QRANT_CLIENT.get_collections().collections]


def check_collection_exists(collection_name):
    return collection_name in get_collection()


@file_handel_router.post("/upload_file")
async def read_item(session_id_or_name: str, item: UploadFile = File(...)):
    session_id_or_name = await return_id(session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}

    if item.content_type != "application/pdf":
        return {"status": "bad", "message": "accpect only pdf"}

    with open(DOCUMENT_SOURCE_DIR + item.filename, "wb") as f:
        f.write(await item.read())

    threading.Thread(target=get_imge_ids,args=(DOCUMENT_SOURCE_DIR + item.filename,)).start()

    file_id = await load_file(DOCUMENT_SOURCE_DIR + item.filename)
    cursor.execute(
        "INSERT INTO file_records(uuid,file_name,file_id) VALUES(?,?,?)",
        (session_id_or_name, item.filename, file_id),
    )
    connection.commit()
    return {"status": "ok", "message": "file inserted"}


def get_hash(file_path: str):
    with open(file_path, "rb") as f:
        file_contents = f.read()

    return hashlib.sha256(file_contents).hexdigest()


def get_imge_ids(src: str):

    reader = PdfReader(src)
    for page_no, page in enumerate(reader.pages):

        for img in page.images:
            hashs = hashlib.sha256(img.data).hexdigest()
            with open(TMP_DIR + str(hashs) + ".png", "wb") as fp:
                fp.write(img.data)
            print(hashs)
            cursor.execute(
                "INSERT OR REPLACE into image_record(image_id,page_no,doc) VALUES(?,?,?)",
                (str(hashs), page_no+1, src),
            )

    connection.commit() 
    print("IMAGE PROCESSED")


def desc_img(file_path):

    pil_image = Image.open(file_path)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # You can change the format if needed
    image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return IMAGE_MODE.bind(images=[image_b64]).invoke("describe the image")


async def load_file(pdf_file_name):

    collection_name = get_hash(pdf_file_name)

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
        url=QDRANT_URL,
        documents=document,
        embedding=embedding_function,
        collection_name=collection_name,
    )

    return collection_name


def text_process(message: str) -> str:
    return GoogleTranslator(source="auto", target="en").translate(message.strip())


@file_handel_router.get("/list_files")
async def read_item(session_id_or_name):
    if await return_id(session_id_or_name) == None:
        return {"status": "bad", "message": "session not exits"}

    collection_list = cursor.execute(
        "SELECT fr.file_id,fr.file_name FROM  file_records fr JOIN user_session us ON fr.uuid = us.uuid"
    ).fetchall()

    return collection_list


@file_handel_router.get("/delete_file")
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


@file_handel_router.get("/delete_all_file")
async def delete_file(session_id_or_name):
    session_id_or_name = await return_id(session_id_or_name)
    if await return_id(session_id_or_name) == None:
        return {"status": "bad", "message": "session not exits"}

    cursor.execute(
        "DELETE FROM file_records WHERE uuid = ?",
        (session_id_or_name,),
    )
    connection.commit()
    return {"status": "Done"}
