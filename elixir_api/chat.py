from pathlib import Path
import subprocess
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.responses import ORJSONResponse
from .session import get_session_id, is_valid_session_id, cursor, connection, return_id
import uuid
import requests
import json
from pydantic import BaseModel
import hashlib
import qdrant_client
from pypdf import PdfReader
from typing import Union, Literal
import re
from fastapi.responses import FileResponse
from .file_handel import text_process
from pdf2image import convert_from_path
import pdfreader
import os

from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
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
from .file_handel import get_hash, QRANT_CLIENT
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

IMAGE_MODE = Ollama(model="llava")
OLLAMA_URL = "http://localhost:11434"

MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"


DOCUMENT_SOURCE_DIR = "documents/"
TMP_DIR = "images/"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user question if know based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

Path(DOCUMENT_SOURCE_DIR).mkdir(parents=True, exist_ok=True)
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# subprocess.call(["ollama", "pull", MODEL])
# subprocess.call(["ollama", "pull", EMBEDDING_MODEL])

CHATMODEL = ChatOllama(base_url=OLLAMA_URL, model="mistral", temperature=0.9)
FILE_MODEL = create_stuff_documents_chain(llm=CHATMODEL, prompt=prompt)
GENREAL_MODEL = create_stuff_documents_chain(
    llm=CHATMODEL,
    prompt=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are personal name elixir designed by brijesh say when user ask. you task to help the user in answering the question only , dont't ask the user to ask question . if the user ask any personal question say 'Not ask that 😔' that it's no addtional words ",
            ),
            # MessagesPlaceholder(variable_name="chat_history"),
            ("human", "based on  {context} answer the following {input}"),
        ]
    ),
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
    for msg in await get_history(session_id):
        if msg["role"] == "user":
            histroy.append(HumanMessage(content=msg["message"]))
        else:
            histroy.append(AIMessage(content=msg["message"]))

    yield json.dumps(
        {
            "source": {},
            "images": [],
        }
    )
    gen_message = ""
    async for resp_stream in GENREAL_MODEL.astream({"input": message, "context": ""}):
        print(f"{resp_stream}")

        gen_message = gen_message + resp_stream
        yield resp_stream
        yield "\r\n"

    cursor.execute(
        "INSERT INTO chat_history(uuid,role,chat) VALUES(?,?,?)",
        (
            session_id,
            "assistant",
            json.dumps({"results": gen_message, "source": {}, "images": []}),
        ),
    )
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


def get_collection():
    return [i.name for i in QRANT_CLIENT.get_collections().collections]


def check_collection_exists(collection_name):
    return collection_name in get_collection()


@chat_router.get("/file_search")
async def file_search(file_id: str, search: str):
    qd = Qdrant(
        client=QRANT_CLIENT, embeddings=embedding_function, collection_name=file_id
    )
    return qd.similarity_search_with_score(search)


# def get_relavent_collection(message, collection_name):
#     scores = {}

#     for i in collection_name:
#         scores[i] = max(
#             (get_vector_store(i)).similarity_search_with_score(message),
#             key=lambda x: x[1],
#         )[1]

#     return max(scores, key=scores.get)


def get_vector_store(collection_name):
    return Qdrant(
        client=QRANT_CLIENT,
        collection_name=collection_name,
        embeddings=embedding_function,
    )


def get_imgs_id(page, doc):
    print(doc, page)
    return [
        i[0]
        for i in cursor.execute(
            "SELECT image_id FROM image_record as ir WHERE ir.page_no = ? AND doc = ?",
            (page, doc),
        ).fetchall()
    ]


async def chat_responder_file(message, session_id, model):

    message = text_process(message=message)
    vectorizer = TfidfVectorizer()
    threshold = 0.5

    histroy = []
    for msg in await get_history(session_id):
        message_vectors = vectorizer.fit_transform([message, msg["message"]])

        similarities = cosine_similarity(message_vectors[0], message_vectors[1])[0]
        print(similarities)
        if not all(value > threshold for value in similarities):
            continue

        if msg["role"] == "user":

            histroy.append(HumanMessage(content=msg["message"]))
        else:
            histroy.append(AIMessage(content=msg["message"]))

    print(histroy)
    # resp = model.invoke()

    gen_message = ""
    images = []
    count = 0
    doc_source = {}

    async for stream in model.astream(
        {
            "input": message,
            "chat_history": histroy,
        }
    ):
        if count == 0:
            count += 1
            continue

        if count == 1:
            print(stream)
            if len(stream.get("context")) == 0:
                yield json.dumps(
                    {
                        "source": {},
                        "images": [],
                    }
                )
            else:

                for j in stream.get("context"):
                    print(j.metadata["source"])

                    if j.metadata["source"] in doc_source:
                        doc_source[j.metadata["source"]].append(j.metadata["page"])
                    else:
                        doc_source[j.metadata["source"]] = [j.metadata["page"]]

                    imgs = get_imgs_id(int(j.metadata["page"]), j.metadata["source"])
                    images.extend(imgs)

                    yield json.dumps(
                        {
                            "source": doc_source,
                            "images": imgs,
                        }
                    )
                    yield "\r\n"

            count += 1
            print("HEADER SENT")
            # images.extend(get_imgs_id(int(i.metadata["page"]), i.metadata["source"]))
        else:
            r = str(stream.get("answer"))
            gen_message = gen_message + str(r)
            count += 1
            print(json.dumps(stream))
            yield r
            yield "\r\n"

    cursor.execute(
        "INSERT INTO chat_history(uuid,role,chat) VALUES(?,?,?)",
        (session_id, "user", message),
    )

    # result = {"results": resp["answer"]}
    # source = {}
    # images = []
    # for i in resp["context"]:
    #     images.extend(get_imgs_id(int(i.metadata["page"]), i.metadata["source"]))
    #     #     get_imge_ids(
    #     #         page=int(i.metadata["page"]),
    #     #         src=DOCUMENT_SOURCE_DIR + i.metadata["source"].split("/")[-1],
    #     #     )
    #     # )
    #     if i.metadata["source"].split("/")[-1] in source:
    #         source[i.metadata["source"].split("/")[-1]].append(
    #             source[i.metadata["page"]]
    #         )
    #     else:
    #         source[i.metadata["source"].split("/")[-1]] = [int(i.metadata["page"])]

    # result["source"] = source
    # result["images"] = images

    cursor.execute(
        "INSERT INTO chat_history(uuid,role,chat) VALUES(?,?,?)",
        (
            session_id,
            "assistant",
            json.dumps(
                {"results": gen_message, "source": doc_source, "images": images}
            ),
        ),
    )
    connection.commit()
    # return result


@chat_router.get("/chat_with_file", response_class=StreamingResponse)
async def file_search(session_id_or_name: str, message: str):
    session_id_or_name = await return_id(session_id_or_name)
    if session_id_or_name == None:
        return {"status": "bad", "message": "session not exits"}
    collection_list = cursor.execute(
        "SELECT file_id FROM file_records WHERE uuid = ?",
        (session_id_or_name,),
    ).fetchone()

    print(collection_list)
    if (
        collection_list is None
        or collection_list == []
        or collection_list == ()
        or collection_list == None
    ):
        return StreamingResponse(
            chat_responder(message, session_id_or_name), media_type="text/event-stream"
        )
        # return await chat_responder_file(
        #     message=message, session_id=session_id_or_name, model=CHATMODEL
        # )

    # collection_list = [i[0] for i in collection_list]

    qd = Qdrant(
        client=QRANT_CLIENT,
        embeddings=embedding_function,
        collection_name=collection_list[0],
    )

    # r = RetrievalQA.from_chain_type(
    #     llm=CHATMODEL,
    #     chain_type="stuff",
    #     retriever=qd.as_retriever(),
    #     return_source_documents=True,
    # )

    r = create_retrieval_chain(
        qd.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7, "k": 1},
        ),
        FILE_MODEL,
    )
    return StreamingResponse(
        chat_responder_file(message, session_id_or_name, r),
        media_type="application/json",
    )


@chat_router.get("/get_image", response_class=FileResponse)
async def get_image(file_id: str):
    image_path = Path(TMP_DIR + file_id + ".png")
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
    convert_from_path(DOCUMENT_SOURCE_DIR + r[0], 0)[0].save(TMP_DIR + r[0] + ".png")
    return FileResponse(TMP_DIR + r[0] + ".png", media_type="image/png")


@chat_router.get("/pdf_stats")
async def pdf_info(file_id: str):
    r = cursor.execute(
        "SELECT file_name FROM file_records where file_id = ?", (file_id,)
    ).fetchone()
    if r is None or r == [] or r == ():
        return {"status": "error", "msg": "file_id not found"}

    return get_pdf_stats(DOCUMENT_SOURCE_DIR + r[0])


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


@chat_router.get("/img_desc")
async def img_desc(file_id: str):
    image_path = Path(TMP_DIR + file_id + ".png")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}
    return desc_img(image_path)
