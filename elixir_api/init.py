from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
import qdrant_client
import configparser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
import sqlite3
from pathlib import Path
import logging
from rich.logging import RichHandler
from subprocess import call as command_execute

__CONFIG__ = configparser.ConfigParser()
__CONFIG__.read("elixir.ini")

logging.basicConfig(
    level="INFO", format="%(levelname)s [%(asctime)s] %(message)s", handlers=[RichHandler(show_time=False,show_level=False)]
)
log = logging.getLogger("rich")


__connection__ = sqlite3.connect(__CONFIG__.get("STORAGE", "SQLITE_FILE"), check_same_thread=False)

def create_tables():
    cursor = __connection__.cursor()
    cursor.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS user_session ( 
                uuid varchar(36) PRIMARY KEY, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                session_name varchar(250)
            );
        CREATE TABLE IF NOT EXISTS chat_history ( 
                uuid varchar(36) REFERENCES user_session(uuid) ON DELETE CASCADE, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL, 
                chat BOLB , 
                role VARCHAR(25) DEFAULT "user" NOT NULL
            );
        CREATE TABLE IF NOT EXISTS file_records ( 
                uuid varchar(36) REFERENCES user_session(uuid) ON DELETE CASCADE UNIQUE, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL, 
                file_name varchar(250) NOT NULL, 
                file_id varchar(64) NOT NULL
            );
        CREATE TABLE IF NOT EXISTS image_record ( 
                image_id VARCHAR(64) PRIMARY KEY, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL, 
                desc BLOB , 
                page_no INT , 
                doc VARCHAR(250) 
            );
        """
    )
    __connection__.commit()



log.info("initalization started")
__embedding_function__ = OllamaEmbeddings(model=__CONFIG__.get("OLLAM", "EMBEDDING_MODEL"))
__qdrant_clinet__ = qdrant_client.QdrantClient(url=__CONFIG__.get("STORAGE", "QDRANT_URL"))


__chat_prompt__ = ChatPromptTemplate.from_messages(
    [
        ("system", __CONFIG__.get("ELIXIR", "SYSTEM_PROMT")),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

__chat_model__ = ChatOllama(
    base_url=__CONFIG__.get("OLLAM", "OLLAMA_URL"),
    model=__CONFIG__.get("OLLAM", "CHAT_MODEL"),
    temperature=float(__CONFIG__.get("OLLAM", "CHAT_MODEL_TEMPERATURE")),
)

__chat_retrival_model__ =  create_stuff_documents_chain(llm=__chat_model__, prompt=__chat_prompt__)

Path(__CONFIG__.get("FOLDERS", "DOCUMENT_SOURCE_DIR")).mkdir(parents=True, exist_ok=True)
Path(__CONFIG__.get("FOLDERS", "IMAGES_DIR")).mkdir(parents=True, exist_ok=True)

command_execute(["ollama", "pull", __CONFIG__.get("OLLAM", "CHAT_MODEL")])
command_execute(["ollama", "pull", __CONFIG__.get("OLLAM", "EMBEDDING_MODEL")])


# GENREAL_MODEL = create_stuff_documents_chain(
#     llm=CHATMODEL,
#     prompt=ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are personal name elixir designed by brijesh say when user ask. you task to help the user in answering the question only , dont't ask the user to ask question . if the user ask any personal question say 'Not ask that 😔' that it's no addtional words ",
#             ),
#             # MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "based on  {context} answer the following {input}"),
#         ]
#     ),
# )


create_tables()
log.info("initalization ended")
