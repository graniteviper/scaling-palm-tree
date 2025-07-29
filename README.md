# 📄 Scaling Palm Tree

A lightweight, CPU-efficient pipeline that semantically extracts and ranks the most relevant content from a set of PDF documents. This solution is specifically tuned for tasks like **travel planning**, with support for contextual understanding using sentence embeddings.

---

## 🚀 Overview

This project focuses on extracting relevant content from travel-related PDF documents for the South of France. It ranks text sections based on their semantic similarity to a given context — emphasizing **gastronomy** and **historical sites** — using a sentence-transformer model.

### 🔍 Features
- Semantic search using `all-MiniLM-L6-v2`
- PDF parsing using `PyMuPDF`
- Cosine similarity-based ranking
- Robust input/output pipeline with error handling
- Compact model footprint (<100MB), CPU-only

---

## 📁 Directory Structure

scaling-palm-tree/

├── approach_explanation.md      

├── Dockerfile                  

├── extractor.py                

├── input.json                   

├── main.py                      

├── output.json                  

├── requirements.txt             

0. Make sure you have docker installed.

    Mac: https://youtu.be/gcacQ29AjOo?si=wx9Bo0TkpyM5nMcE

    Windows: https://youtu.be/JBEUKrjbWqg?si=qLaanGPwb5vK598U

1. to build the image: `docker build --platform linux/amd64 -t mysolution:<your-tag> .`
Replace <your-tag> with any name like v1, v2, etc.

2. For windows: 
``` 
  docker run --rm `
  -v "${PWD}\input:/app/input" `
  -v "${PWD}\output:/app/output" `
  mysolution:<your-tag>
```
   b. For mac:
   ``` 
   docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolution:<your-tag>
```
