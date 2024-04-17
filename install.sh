curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
ollama pull llama2
ollama pull  llava
ollama pull  all-minilm 
ollama pull nomic-embed-text 
ollama pull mxbai-embed-large 

poetry update

sudo docker run -p 6333:6333 -v embedings:/qdrant/storage   qdrant/qdrant&
OLLAMA_HOST=0.0.0.0:2563 ollama serve&
poetry run python uvicorn elixir_api.main:app --host 0.0.0.0 --port 8000 --workers 1 
