from fastapi import APIRouter
import subprocess

settings_router = APIRouter()


DEFAULT_MODEL = "mistral"


@settings_router.get("/set_default_model")
def set_default_model(model_name: str):
    global DEFAULT_MODEL
    DEFAULT_MODEL = model_name


@settings_router.get("/get_default_model")
def set_default_model(model_name: str):
    return {"default_model": MODEL}


@settings_router.get("/avaliable_models")
def avaliable_models():
    r = subprocess.run(["ollama", "ls"], capture_output=True, text=True)
    res = {}
    for i in r.stdout.split("\n")[1:-1]:
        i = i.split("\t")
        res[i[0].strip()] = i[2].strip()
    return res
