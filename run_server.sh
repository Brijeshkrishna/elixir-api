#!/bin/sh


poetry update
ollama serve
sudo docker run -p 6333:6333 -v ./embedings:/qdrant/storage   qdrant/qdrant
