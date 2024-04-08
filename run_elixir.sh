#!/bin/sh


poetry update
ollama serve
poetry run uvicorn elixir_api.main:app --reload