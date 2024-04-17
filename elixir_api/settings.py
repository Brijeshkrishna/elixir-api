from fastapi import APIRouter
import subprocess

settings_router = APIRouter()


@settings_router.get("/avaliable_models")
def avaliable_models():
    r = subprocess.run(["ollama", "ls"], capture_output=True, text=True)
    res = {}
    for i in r.stdout.split("\n")[1:-1]:
        i = i.split("\t")
        res[i[0].strip()] = i[2].strip()
    return res
