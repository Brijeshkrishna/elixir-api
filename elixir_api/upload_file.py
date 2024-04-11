from fastapi import APIRouter, File, UploadFile
from typing import Union
from .session import cursor, return_id, connection
import uuid

history_router = APIRouter()