import json
import time
import os
import numpy as np
import pdfplumber
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class DocumentProcessor:
    def __init__(self):
        # Load the small CPU-friendly model
        self.model = SentenceTransformer('offline_model')

    def extract_outline(self, pdf_path: str) -> dict:
        """Extract document outline with headings and structure"""
        title = ""
        headings = []

        with pdfplumber.open(pdf_path) as pdf:
            # Extract title from first page
            first_page = pdf.pages[0]
            title_candidates = []

            words = first_page.extract_words(extra_attrs=["size", "fontname"], keep_blank_chars=False)
            lines = defaultdict(list)
            for word in words:
                top_key = round(word['top'])
                lines[top_key].append(word)

            for top in sorted(lines.keys()):
                line_words = lines[top]
                min_x0 = min(word['x0'] for word in line_words)
                max_x1 = max(word['x1'] for word in line_words)
                text = " ".join(word['text'] for word in line_words)
                avg_size = np.mean([word['size'] for word in line_words])

                if avg_size > 20 and (min_x0 < 100 and max_x1 > first_page.width - 100):
                    title_candidates.append((avg_size, text))

            if title_candidates:
                title_candidates.sort(key=lambda x: x[0], reverse=True)
                title = title_candidates[0][1]

            heading_candidates = []
            for page_number, page in enumerate(pdf.pages, start=1):
                words = page.extract_words(extra_attrs=["size", "fontname"], keep_blank_chars=False)
                lines = defaultdict(list)
                for word in words:
                    top_key = round(word['top'])
                    lines[top_key].append(word)

                for top in sorted(lines.keys()):
                    line_words = lines[top]
                    min_x0 = min(word['x0'] for word in line_words)
                    max_x1 = max(word['x1'] for word in line_words)
                    text = " ".join(word['text'] for word in line_words)
                    max_size = max(word['size'] for word in line_words)

                    if (max_size > 10 and min_x0 > 50 and max_x1 < page.width - 50 and
                        len(text) < 200 and len(line_words) < 10):
                        heading_candidates.append({
                            "text": text,
                            "size": max_size,
                            "page": page_number,
                            "y": top
                        })

            if heading_candidates:
                sizes = [c['size'] for c in heading_candidates]
                distinct_sizes = sorted(set(sizes), reverse=True)[:3]

                level_map = {}
                for i, size in enumerate(distinct_sizes):
                    level_map[size] = f"H{i+1}"

                current_level = 0
                for candidate in sorted(heading_candidates, key=lambda x: (x['page'], x['y'])):
                    level = level_map.get(candidate['size'], None)
                    if level:
                        level_num = int(level[1:])
                        if level_num <= current_level + 1:
                            headings.append({
                                "level": level,
                                "text": candidate['text'],
                                "page": candidate['page']
                            })
                            current_level = level_num

        return {
            "title": title,
            "outline": headings
        }

    def extract_page_text(self, pdf_path: str, page_number: int) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number <= len(pdf.pages):
                    page = pdf.pages[page_number - 1]
                    return page.extract_text() or ""
            return ""
        except Exception as e:
            print(f"Error extracting text from {pdf_path} page {page_number}: {str(e)}")
            return ""

    def semantic_rank(self, query: str, texts: List[str]) -> np.ndarray:
        query_embed = self.model.encode([query])
        doc_embeds = self.model.encode(texts)
        return cosine_similarity(query_embed, doc_embeds)[0]

    def process_documents(self, input_data: dict) -> dict:
        start_time = time.time()

        challenge_info = input_data.get("challenge_info", {})
        documents = input_data.get("documents", [])
        persona = input_data.get("persona", {}).get("role", "")
        job = input_data.get("job_to_be_done", {}).get("task", "")

        context = f"{persona} performing: {job}"

        if not persona or not job:
            return {
                "error": "Missing persona or job in input data",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }

        all_headings = []
        for doc in documents:
            filename = doc.get("filename")
            if filename:
                pdf_path = os.path.join("input", filename)
                if os.path.exists(pdf_path):
                    outline = self.extract_outline(pdf_path)
                    for heading in outline.get("outline", []):
                        all_headings.append({
                            "document": filename,
                            "title": outline.get("title", ""),
                            "heading": heading["text"],
                            "level": heading["level"],
                            "page": heading["page"],
                            "text": ""
                        })

        if not all_headings:
            return {
                "error": "No headings could be extracted from documents",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }

        heading_texts = [f"{h['title']} - {h['heading']}" for h in all_headings]
        scores = self.semantic_rank(context, heading_texts)

        for i, heading in enumerate(all_headings):
            heading["score"] = float(scores[i])

        ranked_headings = sorted(all_headings, key=lambda x: x["score"], reverse=True)

        for heading in ranked_headings[:10]:
            pdf_path = os.path.join("input", heading["document"])
            heading["text"] = self.extract_page_text(pdf_path, heading["page"])

        output = {
            "metadata": {
                "challenge_info": challenge_info,
                "persona": persona,
                "job": job,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "processing_time_sec": round(time.time() - start_time, 2)
            },
            "results": {
                "top_sections": self._format_sections(ranked_headings[:10]),
                "key_passages": self._format_subsections(ranked_headings[:5])
            }
        }

        return output

    def _format_sections(self, headings: list) -> list:
        return [{
            "document": os.path.basename(h["document"]),
            "page": h["page"],
            "section_title": h["heading"],
            "relevance_score": h["score"],
            "document_title": h["title"]
        } for h in headings]

    def _format_subsections(self, headings: list) -> list:
        return [{
            "document": os.path.basename(h["document"]),
            "page": h["page"],
            "refined_text": h["text"][:1000] + ("..." if len(h["text"]) > 1000 else ""),
            "relevance_score": h["score"],
            "section_title": h["heading"]
        } for h in headings]

def ensure_output_directory():
    os.makedirs("output", exist_ok=True)

def main():
    ensure_output_directory()

    input_path = os.path.join("input", "input.json")
    output_path = os.path.join("output", "output.json")

    try:
        with open(input_path, "r") as f:
            input_data = json.load(f)
    except FileNotFoundError:
        error_output = {"error": f"Input file not found: {input_path}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(output_path, "w") as f:
            json.dump(error_output, f, indent=2)
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError as e:
        error_output = {"error": f"Invalid JSON: {str(e)}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(output_path, "w") as f:
            json.dump(error_output, f, indent=2)
        print(f"Error: Invalid JSON in input file")
        return

    processor = DocumentProcessor()
    result = processor.process_documents(input_data)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Success! Output saved to {output_path}")

if __name__ == "__main__":
    main()
