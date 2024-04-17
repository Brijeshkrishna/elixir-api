from pydantic import BaseModel
from .init import __qdrant_clinet__
import hashlib
from deep_translator import GoogleTranslator
import threading
from pypdf import PdfReader
import io
from PIL import Image
import json
from typing import List
from .init import __connection__, __embedding_function__, log
import configparser
import uuid
from typing import Literal
from langchain_community.vectorstores import Qdrant
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .history import get_history
from langchain_core.messages import HumanMessage, AIMessage


__CONFIG__ = configparser.ConfigParser()
__CONFIG__.read("elixir.ini")

cursor = __connection__.cursor()


class FileUploadBase(BaseModel):
    session_id_or_name: str
    filename: str
    file: str


def get_collection():
    return [i.name for i in __qdrant_clinet__.get_collections().collections]


def check_collection_exists(collection_name):
    return collection_name in get_collection()


def get_hash(file_path: str):
    with open(file_path, "rb") as f:
        file_contents = f.read()
    return hashlib.sha256(file_contents).hexdigest()


def text_process(message: str) -> str:
    return GoogleTranslator(source="auto", target="en").translate(message.strip())


def run_in_thread(f):
    def wrapper(*args, **kwargs):
        threading.Thread(target=f, args=args, kwargs=kwargs).start()

    return wrapper


def parse_images_from_pdf(file_id: str):
    src = get_file_name(file_id)
    reader = PdfReader(src)
    for page_no, page in enumerate(reader.pages):

        for img in page.images:
            hashs = hashlib.sha256(img.data).hexdigest()
            with open(
                __CONFIG__.get("FOLDERS", "IMAGES_DIR") + str(hashs) + ".png", "wb"
            ) as fp:
                fp.write(img.data)
            cursor.execute(
                "INSERT OR REPLACE into image_record(image_id,page_no,doc) VALUES(?,?,?)",
                (str(hashs), page_no + 1, src),
            )

    commit()


def image_to_base64(filepath):
    pil_image = Image.open(file_path)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())


def inserted_db(session_id: uuid.UUID, role: Literal["user", "assistant"], message):
    cursor.execute(
        "INSERT INTO chat_history(uuid,role,chat) VALUES(?,?,?)",
        (session_id, role, message),
    )


async def collect_history(message, session_id):
    vectorizer = TfidfVectorizer()

    histroy = []
    for msg in await get_history(session_id):

        message_vectors = vectorizer.fit_transform([message, msg["message"]])
        similarities = cosine_similarity(message_vectors[0], message_vectors[1])[0]
        log.info(
            f"message {msg['message']} as history score {sum(similarities) / len(similarities  )}"
        )

        if not all(
            value > float(__CONFIG__.get("ELIXIR", "HISTORY_RELATION_THRESHOLD"))
            and value <= 0.9
            for value in similarities
        ):
            continue
        log.info(
            f"message {msg['message']} as history score {sum(similarities) / len(similarities  )}"
        )
        if msg["role"] == "user":
            histroy.append(HumanMessage(content=msg["message"]))
        else:
            histroy.append(AIMessage(content=msg["message"]))
    return histroy


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
        client=__qdrant_clinet__,
        collection_name=collection_name,
        embeddings=__embedding_function__,
    )


def get_image_id(page, doc):
    return [
        i[0]
        for i in cursor.execute(
            "SELECT image_id FROM image_record as ir WHERE ir.page_no = ? AND doc = ?",
            (page, doc),
        ).fetchall()
    ]


def commit():
    __connection__.commit()


def terminal() -> str:
    return "\r\n"


def get_file_name(file_id):
    return __CONFIG__.get("FOLDERS", "DOCUMENT_SOURCE_DIR") + str(
        cursor.execute(
            "SELECT file_name FROM file_records WHERE file_id = ?", (file_id,)
        ).fetchone()[0]
    )
