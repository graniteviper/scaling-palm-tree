# Approach Explanation

## Objective

The goal of this solution is to extract and rank the most relevant content from a collection of PDF documents based on a specific context. In this challenge, the context is focused on *travel planning for the South of France, with an emphasis on **gastronomy and historical sites*. The solution identifies which sections of the PDFs are most useful for this purpose by using semantic similarity.

## Methodology

The approach is composed of the following major components:

### 1. PDF Text Extraction

We use the PyMuPDF (fitz) library to open and read PDF files. Each page is processed to extract raw text using the get_text("text") method. Only non-empty pages are retained, and each extracted text block is stored along with metadata such as the filename and page number. This forms the basic input for semantic processing.

### 2. Semantic Embedding

To capture the meaning of both the context query and the PDF content, we use the all-MiniLM-L6-v2 model from sentence-transformers. This model converts text into fixed-size vector embeddings that capture semantic information. The context and each extracted block of text are transformed into embeddings, enabling meaningful comparison.

### 3. Similarity Scoring

We compute the *cosine similarity* between the context vector and each block’s vector to measure how closely each piece of content aligns with the travel-related context. A higher score means greater semantic relevance. This scoring is efficient and can handle numerous documents in a scalable way.

### 4. Ranking and Selection

The blocks are sorted in descending order based on their similarity scores. The top 10 blocks are formatted into a high-level summary (extracted_sections) that includes document names, page numbers, and relevance scores. A deeper analysis (sub_section_analysis) is provided for the top 5 blocks, showing a truncated preview of the actual text.

### 5. Output Format

The final output is a structured JSON containing:
- Metadata (e.g., timestamps, processing time, filenames)
- Top-ranked document sections
- Refined excerpts from the top pages for closer inspection

### 6. Robust Error Handling

The system gracefully handles common issues like missing files, invalid file paths, and malformed JSON input. All errors are logged in the output, ensuring traceability and smoother debugging.

## Benefits

This pipeline is modular, lightweight, and scalable. It is designed for *CPU-only environments* and maintains a small memory footprint (<100MB). The model-agnostic design allows for easy upgrades to more powerful models if needed. Additionally, this approach avoids reliance on unreliable heuristics like font sizes or hardcoded rules, instead using *true semantic understanding* to determine relevance.