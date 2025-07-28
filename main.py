# import json
# import time
# import os
# import numpy as np
# import pdfplumber
# import re
# from collections import defaultdict
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import List, Dict, Any

# class DocumentProcessor:
#     def _init_(self):
#         # Load the small CPU-friendly model
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     def extract_outline(self, pdf_path: str) -> dict:
#         """Extract document outline with headings and structure"""
#         title = ""
#         headings = []
        
#         with pdfplumber.open(pdf_path) as pdf:
#             # Extract title from first page
#             first_page = pdf.pages[0]
#             title_candidates = []
            
#             # Process words in reading order
#             words = first_page.extract_words(
#                 extra_attrs=["size", "fontname"],
#                 keep_blank_chars=False
#             )
            
#             # Group words by line
#             lines = defaultdict(list)
#             for word in words:
#                 top_key = round(word['top'])
#                 lines[top_key].append(word)
            
#             # Find title candidate (top-most large text)
#             for top in sorted(lines.keys()):
#                 line_words = lines[top]
#                 min_x0 = min(word['x0'] for word in line_words)
#                 max_x1 = max(word['x1'] for word in line_words)
#                 text = " ".join(word['text'] for word in line_words)
#                 avg_size = np.mean([word['size'] for word in line_words])
                
#                 # Consider as title candidate if large and centered
#                 if avg_size > 20 and (min_x0 < 100 and max_x1 > first_page.width - 100):
#                     title_candidates.append((avg_size, text))
            
#             if title_candidates:
#                 title_candidates.sort(key=lambda x: x[0], reverse=True)
#                 title = title_candidates[0][1]
            
#             # Extract headings from all pages
#             heading_candidates = []
#             for page_number, page in enumerate(pdf.pages, start=1):
#                 words = page.extract_words(
#                     extra_attrs=["size", "fontname"],
#                     keep_blank_chars=False
#                 )
                
#                 # Group words into lines
#                 lines = defaultdict(list)
#                 for word in words:
#                     top_key = round(word['top'])
#                     lines[top_key].append(word)
                
#                 # Process each line
#                 for top in sorted(lines.keys()):
#                     line_words = lines[top]
#                     min_x0 = min(word['x0'] for word in line_words)
#                     max_x1 = max(word['x1'] for word in line_words)
#                     text = " ".join(word['text'] for word in line_words)
#                     max_size = max(word['size'] for word in line_words)
#                     fonts = [word['fontname'] for word in line_words]
                    
#                     # Heading characteristics:
#                     # - Larger than body text
#                     # - Not too close to page edges
#                     # - Not too long
#                     if (max_size > 10 and 
#                         min_x0 > 50 and 
#                         max_x1 < page.width - 50 and
#                         len(text) < 200 and 
#                         len(line_words) < 10):
#                         heading_candidates.append({
#                             "text": text,
#                             "size": max_size,
#                             "page": page_number,
#                             "y": top
#                         })
            
#             # Cluster heading levels
#             if heading_candidates:
#                 sizes = [c['size'] for c in heading_candidates]
#                 distinct_sizes = sorted(set(sizes), reverse=True)[:3]
                
#                 # Assign heading levels
#                 level_map = {}
#                 for i, size in enumerate(distinct_sizes):
#                     level_map[size] = f"H{i+1}"
                
#                 # Create outline with hierarchical validation
#                 current_level = 0
#                 for candidate in sorted(heading_candidates, key=lambda x: (x['page'], x['y'])):
#                     level = level_map.get(candidate['size'], None)
#                     if level:
#                         level_num = int(level[1:])
                        
#                         # Maintain hierarchy (H1 > H2 > H3)
#                         if level_num <= current_level + 1:
#                             headings.append({
#                                 "level": level,
#                                 "text": candidate['text'],
#                                 "page": candidate['page']
#                             })
#                             current_level = level_num
        
#         return {
#             "title": title,
#             "outline": headings
#         }

#     def extract_page_text(self, pdf_path: str, page_number: int) -> str:
#         """Extract text from a specific page"""
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 if page_number <= len(pdf.pages):
#                     page = pdf.pages[page_number - 1]
#                     return page.extract_text() or ""
#             return ""
#         except Exception as e:
#             print(f"Error extracting text from {pdf_path} page {page_number}: {str(e)}")
#             return ""

#     def semantic_rank(self, query: str, texts: List[str]) -> np.ndarray:
#         """Rank texts by similarity to query"""
#         query_embed = self.model.encode([query])
#         doc_embeds = self.model.encode(texts)
#         return cosine_similarity(query_embed, doc_embeds)[0]

#     def process_documents(self, input_data: dict) -> dict:
#         """Main processing function with persona and job context"""
#         start_time = time.time()
        
#         # Extract data from input
#         challenge_info = input_data.get("challenge_info", {})
#         documents = input_data.get("documents", [])
#         persona = input_data.get("persona", "")
#         job = input_data.get("job", "")
        
#         # Prepare search context
#         context = f"{persona} performing: {job}"
        
#         # Validate input
#         if not persona or not job:
#             return {
#                 "error": "Missing persona or job in input data",
#                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
#             }
        
#         # Extract outlines and collect headings from all documents
#         all_headings = []
#         for doc in documents:
#             filename = doc.get("filename")
#             if filename and os.path.exists(filename):
#                 outline = self.extract_outline(filename)
#                 for heading in outline.get("outline", []):
#                     all_headings.append({
#                         "document": filename,
#                         "title": outline.get("title", ""),
#                         "heading": heading["text"],
#                         "level": heading["level"],
#                         "page": heading["page"],
#                         "text": ""  # Will be filled later
#                     })
        
#         if not all_headings:
#             return {
#                 "error": "No headings could be extracted from documents",
#                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
#             }
        
#         # Rank headings by relevance to context
#         heading_texts = [f"{h['title']} - {h['heading']}" for h in all_headings]
#         scores = self.semantic_rank(context, heading_texts)
        
#         # Add scores to headings
#         for i, heading in enumerate(all_headings):
#             heading["score"] = float(scores[i])
        
#         # Sort by relevance
#         ranked_headings = sorted(all_headings, key=lambda x: x["score"], reverse=True)
        
#         # Extract text for top sections
#         for heading in ranked_headings[:10]:
#             heading["text"] = self.extract_page_text(heading["document"], heading["page"])
        
#         # Prepare the output structure
#         output = {
#             "metadata": {
#                 "challenge_info": challenge_info,
#                 "persona": persona,
#                 "job": job,
#                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
#                 "processing_time_sec": round(time.time() - start_time, 2)
#             },
#             "results": {
#                 "top_sections": self._format_sections(ranked_headings[:10]),
#                 "key_passages": self._format_subsections(ranked_headings[:5])
#             }
#         }
        
#         return output
    
#     def _format_sections(self, headings: list) -> list:
#         """Format the top sections"""
#         return [{
#             "document": os.path.basename(h["document"]),
#             "page": h["page"],
#             "section_title": h["heading"],
#             "relevance_score": h["score"],
#             "document_title": h["title"]
#         } for h in headings]
    
#     def _format_subsections(self, headings: list) -> list:
#         """Format the key passages"""
#         return [{
#             "document": os.path.basename(h["document"]),
#             "page": h["page"],
#             "refined_text": h["text"][:1000] + ("..." if len(h["text"]) > 1000 else ""),
#             "relevance_score": h["score"],
#             "section_title": h["heading"]
#         } for h in headings]

# def ensure_output_directory():
#     """Ensure output directory exists"""
#     os.makedirs("output", exist_ok=True)

# def main():
#     ensure_output_directory()
    
#     # File paths
#     input_path = "input.json"
#     output_path = os.path.join("output.json")
    
#     # Try to load input
#     try:
#         with open(input_path, "r") as f:
#             input_data = json.load(f)
#     except FileNotFoundError:
#         error_output = {"error": f"Input file not found: {input_path}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
#         with open(output_path, "w") as f:
#             json.dump(error_output, f, indent=2)
#         print(f"Error: Input file not found at {input_path}")
#         return
#     except json.JSONDecodeError as e:
#         error_output = {"error": f"Invalid JSON: {str(e)}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
#         with open(output_path, "w") as f:
#             json.dump(error_output, f, indent=2)
#         print(f"Error: Invalid JSON in input file")
#         return
    
#     # Process documents
#     processor = DocumentProcessor()
#     result = processor.process_documents(input_data)
    
#     # Save output
#     with open(output_path, "w") as f:
#         json.dump(result, f, indent=2)
    
#     print(f"Success! Output saved to {output_path}")

# if __name__ == "__main__":
#     main()










import json
import time
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from extractor import extract_outline, extract_page_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PersonaDocIntel")

class DocumentProcessor:
    def _init_(self):
        # Load the small CPU-friendly model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded sentence transformer model")
    
    def semantic_rank(self, query: str, texts: list) -> np.ndarray:
        """Rank texts by similarity to query"""
        query_embed = self.model.encode([query])
        doc_embeds = self.model.encode(texts)
        return cosine_similarity(query_embed, doc_embeds)[0]

    def process_documents(self, input_data: dict) -> dict:
        """Main processing function with persona and job context"""
        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Extract data from input
        challenge_info = input_data.get("challenge_info", {})
        documents = input_data.get("documents", [])
        
        # Extract persona and job with fallbacks
        persona = input_data.get("persona") or challenge_info.get("persona") or "Professional"
        job = input_data.get("job") or challenge_info.get("job") or "Analyze documents"
        
        # Prepare search context
        context = f"{persona} performing: {job}"
        logger.info(f"Processing context: {context}")
        
        # Extract outlines and collect headings from all documents
        all_headings = []
        processed_files = []
        for doc in documents:
            filename = doc.get("filename")
            if filename and os.path.exists(filename):
                logger.info(f"Processing document: {filename}")
                try:
                    outline = extract_outline(filename)
                    for heading in outline.get("outline", []):
                        all_headings.append({
                            "document": filename,
                            "title": outline.get("title", os.path.basename(filename)),
                            "heading": heading["text"],
                            "level": heading["level"],
                            "page": heading["page"],
                            "text": ""  # Will be filled later
                        })
                    processed_files.append(os.path.basename(filename))
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
            elif filename:
                logger.warning(f"Document not found: {filename}")
        
        if not all_headings:
            error_msg = "No headings could be extracted from documents"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "timestamp": timestamp,
                "processing_time_sec": round(time.time() - start_time, 2)
            }
        
        # Rank headings by relevance to context
        heading_texts = [f"{h['title']} - {h['heading']}" for h in all_headings]
        scores = self.semantic_rank(context, heading_texts)
        
        # Add scores to headings
        for i, heading in enumerate(all_headings):
            heading["score"] = float(scores[i])
        
        # Sort by relevance
        ranked_headings = sorted(all_headings, key=lambda x: x["score"], reverse=True)
        logger.info(f"Ranked {len(ranked_headings)} headings")
        
        # Extract text for top sections
        for heading in ranked_headings[:10]:
            try:
                heading["text"] = extract_page_text(heading["document"], heading["page"])
            except Exception as e:
                logger.error(f"Error extracting text: {str(e)}")
                heading["text"] = ""
        
        # Prepare the output structure
        output = {
            "metadata": {
                "challenge_info": challenge_info,
                "persona": persona,
                "job": job,
                "timestamp": timestamp,
                "processing_time_sec": round(time.time() - start_time, 2),
                "documents_processed": processed_files
            },
            "results": {
                "top_sections": self._format_sections(ranked_headings[:10]),
                "key_passages": self._format_subsections(ranked_headings[:5])
            }
        }
        
        return output
    
    def _format_sections(self, headings: list) -> list:
        """Format the top sections"""
        return [{
            "document": os.path.basename(h["document"]),
            "page_number": h["page"],
            "section_title": h["heading"],
            "importance_rank": i+1,
            "relevance_score": h["score"]
        } for i, h in enumerate(headings)]
    
    def _format_subsections(self, headings: list) -> list:
        """Format the key passages"""
        return [{
            "document": os.path.basename(h["document"]),
            "page_number": h["page"],
            "refined_text": (h["text"][:1500] + "...") if len(h["text"]) > 1500 else h["text"],
            "importance_rank": i+1,
            "relevance_score": h["score"]
        } for i, h in enumerate(headings)]

def ensure_output_directory():
    """Ensure output directory exists"""
    os.makedirs("output", exist_ok=True)

def main():
    ensure_output_directory()
    logger.info("Starting document processing")
    
    # File paths
    input_path = "input.json"
    output_path = os.path.join("output", "output.json")
    
    # Try to load input
    try:
        with open(input_path, "r") as f:
            input_data = json.load(f)
        logger.info(f"Loaded input from {input_path}")
    except FileNotFoundError:
        error_msg = f"Input file not found: {input_path}"
        logger.error(error_msg)
        error_output = {"error": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(output_path, "w") as f:
            json.dump(error_output, f, indent=2)
        return
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON: {str(e)}"
        logger.error(error_msg)
        error_output = {"error": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(output_path, "w") as f:
            json.dump(error_output, f, indent=2)
        return
    
    # Process documents
    processor = DocumentProcessor()
    result = processor.process_documents(input_data)
    
    # Save output
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()