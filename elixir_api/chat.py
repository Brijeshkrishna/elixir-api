import os
import re
import json
import uuid
import hashlib
import threading
from pathlib import Path
import subprocess
from typing import Union, Literal
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import requests
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse, ORJSONResponse

from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_community.history import get_history
from langchain_community.helper import (
    text_process,
    inserted_db,
    collect_history,
    get_vector_store,
    get_image_id,
    get_hash,
    commit,
    terminal,
    parse_images_from_pdf,
)
from langchain_community.init import (
    __connection__,
    __chat_retrival_model__,
    __embedding_function__,
    __qdrant_clinet__,
    log,
)

from pypdf import PdfReader
import pdfreader
import qdrant_client


__CONFIG__ = configparser.ConfigParser()
__CONFIG__.read("elixir.ini")


chat_router = APIRouter()
cursor = __connection__.cursor()


async def chat_responder(message, session_id):

    message = text_process(message=message)
    inserted_db(session_id=session_id, role="user", message=message)

    # history = await collect_history(message=message, session_id=session_id)

    yield json.dumps(
        {
            "source": {},
            "images": [],
        }
    )

    ai_message = ""
    log.info("chat header sent")
    async for resp_stream in __chat_retrival_model__.astream(
        {"input": message, "context": "", "chat_history": []}
    ):

        ai_message = ai_message + resp_stream
        log.info(f"AI ==>{resp_stream} {[ord(char) for char in resp_stream]} ")
        yield resp_stream
        yield terminal()

    inserted_db(
        session_id=session_id,
        role="assistant",
        message=json.dumps({"results": ai_message, "source": {}, "images": []}),
    )
    commit()
    log.info(f"message saved")


@chat_router.get("/chat")
async def chat(session_id_or_name: Union[str, uuid.UUID], message: str):
    session_id_or_name = await return_id(session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}

    return StreamingResponse(
        chat_responder(message, session_id_or_name), media_type="text/event-stream"
    )


@chat_router.get("/file_search")
async def file_search(collection_name: str, search: str):
    return get_vector_store(
        collection_name=collection_name
    ).similarity_search_with_score(search)


def elixir_header_message(content):
    captured_images = []
    source_doc = {}
    log.info(f"retrived content {content}")
    for j in content.get("context"):

        if j.metadata["source"] in source_doc:
            source_doc[j.metadata["source"]].add(j.metadata["page"])
        else:
            source_doc[j.metadata["source"]] = set([j.metadata["page"]])

        imgs = get_image_id(int(j.metadata["page"]), j.metadata["source"])
        captured_images.extend(imgs)

    return {
        "source": {key: list(value) for key, value in source_doc.items()},
        "images": list(set(captured_images)),
    }


async def chat_responder_file(message, session_id, model):

    message = text_process(message=message)
    history = await collect_history(message=message, session_id=session_id)

    log.info(f"Message after processed {message}")
    log.info(f"History collected len is {len(history)}")

    ai_message = ""

    streams = model.astream(
        {
            "input": message,
            "chat_history": history,
        }
    )

    await anext(streams)  # no need of the fisrt message

    log.info("generating header")
    elixir_head = elixir_header_message(
        await anext(streams)
    )  # process the second message

    yield json.dumps(elixir_head)
    yield "\r\n"  # send the second message

    log.info(
        f"Metadata sent sources:{elixir_head['source']} , images :{elixir_head['images']}"
    )
    async for stream in streams:
        answer = stream.get("answer")
        ai_message = ai_message + answer
        log.info(f"AI ==>{answer} {[ord(char) for char in answer]}")
        yield answer
        yield "\r\n"

    inserted_db(
        session_id=session_id,
        role="assistant",
        message=json.dumps(
            {
                "results": ai_message,
                "source": elixir_head["source"],
                "images": elixir_head["images"],
            }
        ),
    )
    commit()
    log.info(f"message saved")


@chat_router.get("/chat_with_file", response_class=StreamingResponse)
async def chat_with_file(session_id_or_name: str, message: str):

    log.info(f"chat_with_file ( {session_id_or_name}, {message} )")

    session_id_or_name = await return_id(session_id_or_name)
    if session_id_or_name == None:
        log.warn(f"{session_id_or_name} invalid session id")
        return {"status": "bad", "message": "session not exits"}

    collection = cursor.execute(
        "SELECT file_id FROM file_records WHERE uuid = ?",
        (session_id_or_name,),
    ).fetchone()

    log.info(f"file id {collection} of the session id {session_id_or_name}")

    if collection is None or collection == [] or collection == () or collection == None:
        log.warn(f"No collection found retirecting to chat")
        return StreamingResponse(
            chat_responder(message, session_id_or_name), media_type="text/event-stream"
        )

    parse_images_from_pdf(collection[0])

    qd = get_vector_store(collection_name=collection[0])

    r = create_retrieval_chain(
        qd.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": float(__CONFIG__.get("ELIXIR", "SCORE_THRESHOLD")),
                "k": int(__CONFIG__.get("ELIXIR", "NUMBER_OF_DOCUMENTS")),
            },
        ),
        __chat_retrival_model__,
    )
    return StreamingResponse(
        chat_responder_file(message, session_id_or_name, r),
        media_type="application/json",
    )


@chat_router.get("/get_image", response_class=FileResponse)
async def get_image(file_id: str):
    image_path = Path(__CONFIG__.get("FOLDERS", "IMAGES_DIR") + file_id + ".png")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(image_path)


@chat_router.get("/pdf_thumb", response_class=FileResponse)
async def pdf_info(file_id: str):
    r = cursor.execute(
        "SELECT file_name FROM file_records where file_id = ?", (file_id,)
    ).fetchone()
    if r is None or r == [] or r == ():
        return {"status": "error", "msg": "file_id not found"}
    convert_from_path(__CONFIG__.get("FOLDERS", "DOCUMENT_SOURCE_DIR") + r[0], 0)[
        0
    ].save(__CONFIG__.get("FOLDERS", "IMAGES_DIR") + r[0] + ".png")
    return FileResponse(
        __CONFIG__.get("FOLDERS", "IMAGES_DIR") + r[0] + ".png", media_type="image/png"
    )


@chat_router.get("/pdf_stats")
async def pdf_stats(file_id: str):
    r = cursor.execute(
        "SELECT file_name FROM file_records where file_id = ?", (file_id,)
    ).fetchone()
    if r is None or r == [] or r == ():
        return {"status": "error", "msg": "file_id not found"}

    return get_pdf_stats(__CONFIG__.get("FOLDERS", "DOCUMENT_SOURCE_DIR") + r[0])


def get_pdf_stats(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = pdfreader.SimplePDFViewer(file)
        num_pages = 0
        total_words = 0
        total_chars = 0
        images = 10
        for page in reader:
            num_pages += 1
            images = images + len(page.images)
            text = page.text_content
            words = text.split()
            total_words += len(words)
            total_chars += len(text)

        avg_words_per_page = total_words / num_pages
        avg_chars_per_page = total_chars / num_pages

        stats = {
            "size": os.path.getsize(pdf_path),
            "sha256": get_hash(pdf_path),
            "Total number of pages": num_pages,
            "Total number of images": images,
            "Total word count": total_words,
            "Total character count": total_chars,
            "Average words per page": avg_words_per_page,
            "Average characters per page": avg_chars_per_page,
        }

        return stats


# @chat_router.get("/img_desc")
# async def img_desc(file_id: str):
#     image_path = Path(__CONFIG__.get("FOLDERS", "IMAGES_DIR") + file_id + ".png")
#     if not image_path.is_file():
#         return {"error": "Image not found on the server"}
#     return desc_img(image_path)
