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
        try:
            # Load the small CPU-friendly model
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Successfully loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {str(e)}")
            raise e
    
    def semantic_rank(self, query: str, texts: list) -> np.ndarray:
        """Rank texts by similarity to query"""
        if not hasattr(self, 'model') or self.model is None:
            raise AttributeError("Model not properly initialized. Please check _init_ method.")
        
        try:
            query_embed = self.model.encode([query])
            doc_embeds = self.model.encode(texts)
            return cosine_similarity(query_embed, doc_embeds)[0]
        except Exception as e:
            logger.error(f"Error in semantic ranking: {str(e)}")
            raise e

    def process_documents(self, input_data: dict) -> dict:
        """Main processing function with persona and job context"""
        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Extract data from input
        challenge_info = input_data.get("challenge_info", {})
        documents = input_data.get("documents", [])
        
        # Extract persona and job with fallbacks - Handle any role dynamically
        persona = input_data.get("persona")
        if not persona:
            persona = challenge_info.get("persona")
        if not persona:
            persona = "Professional"
        
        # Handle case where persona might be a dict (like {'role': 'Any Role Name'})
        if isinstance(persona, dict):
            if 'role' in persona:
                persona = persona['role']
            else:
                # If it's a dict but no 'role' key, use the first value or stringify
                persona = next(iter(persona.values())) if persona else "Professional"
        
        job = input_data.get("job")
        if not job:
            job = challenge_info.get("job")
        if not job:
            job = "Analyze documents"
        
        # Prepare search context
        context = f"{persona} performing: {job}"
        logger.info(f"Processing context: {context}")
        
        # Extract outlines and collect headings from all documents
        all_headings = []
        processed_files = []
        document_summaries = []
        
        for doc in documents:
            filename = doc.get("filename")
            if filename and os.path.exists(filename):
                logger.info(f"Processing document: {filename}")
                try:
                    outline_result = extract_outline(filename)
                    doc_title = outline_result.get("title", os.path.basename(filename))
                    
                    # Store document summary for metadata
                    document_summaries.append({
                        "filename": os.path.basename(filename),
                        "title": doc_title,
                        "structure": outline_result.get("document_structure", {}),
                        "content_summary": outline_result.get("content_summary", {}),
                        "metadata": outline_result.get("metadata", {})
                    })
                    
                    # Process headings with enhanced information
                    for heading in outline_result.get("outline", []):
                        all_headings.append({
                            "document": filename,
                            "document_title": doc_title,
                            "heading": heading["text"],
                            "level": heading["level"],
                            "page": heading["page"],
                            "font_size": heading.get("font_size", 12),
                            "position": heading.get("position", {}),
                            "heading_metadata": heading.get("metadata", {}),
                            "text": ""  # Will be filled later
                        })
                    
                    processed_files.append(os.path.basename(filename))
                    logger.info(f"Extracted {len(outline_result.get('outline', []))} headings from {os.path.basename(filename)}")
                    
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
        heading_texts = [f"{h['document_title']} - {h['heading']}" for h in all_headings]
        
        try:
            scores = self.semantic_rank(context, heading_texts)
        except Exception as e:
            logger.error(f"Error in semantic ranking: {str(e)}")
            # Fallback: assign random scores or equal scores
            scores = np.random.random(len(heading_texts))
        
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
        
        # Prepare the output structure with enhanced metadata
        output = {
            "metadata": {
                "challenge_info": challenge_info,
                "persona": persona,
                "job": job,
                "timestamp": timestamp,
                "processing_time_sec": round(time.time() - start_time, 2),
                "documents_processed": processed_files,
                "document_summaries": document_summaries,
                "total_headings_found": len(all_headings),
                "semantic_ranking_enabled": hasattr(self, 'model') and self.model is not None
            },
            "results": {
                "top_sections": self._format_sections(ranked_headings[:10]),
                "key_passages": self._format_subsections(ranked_headings[:5]),
                "document_overview": self._format_document_overview(document_summaries),
                "content_distribution": self._format_content_distribution(all_headings)
            }
        }
        
        return output
    
    def _format_sections(self, headings: list) -> list:
        """Format the top sections with enhanced metadata"""
        return [{
            "document": os.path.basename(h["document"]),
            "document_title": h.get("document_title", ""),
            "page_number": h["page"],
            "section_title": h["heading"],
            "heading_level": h["level"],
            "font_size": h.get("font_size", 0),
            "importance_rank": i+1,
            "relevance_score": h["score"],
            "position": h.get("position", {}),
            "metadata": h.get("heading_metadata", {})
        } for i, h in enumerate(headings)]
    
    def _format_subsections(self, headings: list) -> list:
        """Format the key passages with enhanced content"""
        return [{
            "document": os.path.basename(h["document"]),
            "document_title": h.get("document_title", ""),
            "page_number": h["page"],
            "section_title": h["heading"],
            "heading_level": h["level"],
            "refined_text": (h["text"][:1500] + "...") if len(h["text"]) > 1500 else h["text"],
            "text_length": len(h["text"]),
            "importance_rank": i+1,
            "relevance_score": h["score"],
            "extraction_success": bool(h["text"].strip())
        } for i, h in enumerate(headings)]
    
    def _format_document_overview(self, document_summaries: list) -> list:
        """Format document overview information"""
        return [{
            "filename": doc["filename"],
            "title": doc["title"],
            "total_sections": doc["structure"].get("total_headings", 0),
            "structure_quality": doc["structure"].get("structure_quality", "unknown"),
            "key_topics": doc["content_summary"].get("key_topics", [])[:5],
            "content_depth": doc["content_summary"].get("content_depth", "unknown"),
            "pages": doc["metadata"].get("total_pages", 0),
            "file_size_mb": doc["metadata"].get("file_size_mb", 0)
        } for doc in document_summaries]
    
    def _format_content_distribution(self, all_headings: list) -> dict:
        """Analyze content distribution across documents"""
        if not all_headings:
            return {}
        
        # Count by document
        doc_counts = {}
        level_counts = {}
        
        for heading in all_headings:
            doc_name = os.path.basename(heading["document"])
            doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
            level_counts[heading["level"]] = level_counts.get(heading["level"], 0) + 1
        
        return {
            "headings_per_document": doc_counts,
            "heading_levels_distribution": level_counts,
            "total_headings": len(all_headings),
            "documents_with_content": len(doc_counts)
        }

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
    try:
        processor = DocumentProcessor()
        result = processor.process_documents(input_data)
    except Exception as e:
        error_msg = f"Error initializing or processing documents: {str(e)}"
        logger.error(error_msg)
        error_output = {"error": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(output_path, "w") as f:
            json.dump(error_output, f, indent=2)
        return
    
    # Save output
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()