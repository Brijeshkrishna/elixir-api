import chainlit as cl


from pathlib import Path
import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

# from .session import get_session_id, is_valid_session_id, cursor, connection, return_id
import uuid
import requests
import json
from pydantic import BaseModel
import hashlib
import qdrant_client
import sqlite3

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

model = ChatOllama(base_url=OLLAMA_URL, model="mistral", temperature=1)

embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)

connection = sqlite3.connect("history.sqlite", check_same_thread=False)
cursor = connection.cursor()


# @cl.on_chat_start
# async def start():
#     text_content = "# Hello, this is a text element."
#     elements = [cl.Text(name="simple_text", content=text_content, display="inline")]

#     await cl.Message(
#         content="# Check out this text element!",
#         elements=elements,
#     ).send()


@cl.on_chat_start
async def start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["application/pdf"]
        ).send()

    text_file = files[0]

    msg = cl.Message(content="")
    await msg.send()

    await cl.sleep(1)
    with open(text_file.path, "rb") as f:
        file_contents = f.read()

    collection_name = hashlib.sha256(file_contents).hexdigest()
    if check_collection_exists(collection_name):
        msg.content = "File exist, you are ready to chat now"
        await msg.update()
        cl.user_session.set("file_id",collection_name)
        return

    msg.content = f"started processing the pdf"
    await msg.update()

    document = PyPDFLoader(text_file.path, extract_images=False).load_and_split()

    # db = Qdrant(
    #     client=QRANT_CLIENT,
    #     collection_name=collection_name,
    #     embeddings=embedding_function,
    # )
    # print(db)
    # db.add_documents(document)

    await cl.sleep(5)

    msg.content = f"started embedding the file it will take a while"
    await msg.update()

    await cl.sleep(10)
    Qdrant.from_documents(
        url=QDRANT_URL,
        documents=document,
        embedding=embedding_function,
        collection_name=collection_name,
    )
    cl.user_session.set("file_id",collection_name)

    msg.content = f"you are ready to chat now"
    await msg.send()


@cl.on_message
async def on_message(message: cl.Message):

    msg = cl.Message(content="")
    await msg.send()

    await cl.sleep(2)

    qd = Qdrant(
        client=QRANT_CLIENT,
        embeddings=embedding_function,
        collection_name=cl.user_session.get("file_id"),
    )

    r = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=qd.as_retriever(),
        return_source_documents=True,
    )
    print("loaded")
    msg.content = r(message.content)
    await msg.send()
    
    # for i in r.stream(message.content):
    #     msg.content = msg.content + i.content
    #     await msg.update()

    # await msg.send()



def get_collection():
    return [i.name for i in QRANT_CLIENT.get_collections().collections]


def check_collection_exists(collection_name):
    return collection_name in get_collection()







