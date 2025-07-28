
import pdfplumber
import numpy as np
from collections import defaultdict
import re
import logging

logger = logging.getLogger(__name__)

def extract_outline(pdf_path: str) -> dict:
    """Extract document structure including title and headings"""
    title = ""
    headings = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract title from first page
            if len(pdf.pages) > 0:
                first_page = pdf.pages[0]
                title_candidates = []
                
                # Process words in reading order
                words = first_page.extract_words(
                    extra_attrs=["size", "fontname"],
                    keep_blank_chars=False
                )
                
                # Group words by line
                lines = defaultdict(list)
                for word in words:
                    top_key = round(word['top'])
                    lines[top_key].append(word)
                
                # Find title candidate
                for top in sorted(lines.keys()):
                    line_words = lines[top]
                    if not line_words:
                        continue
                        
                    min_x0 = min(word['x0'] for word in line_words)
                    max_x1 = max(word['x1'] for word in line_words)
                    text = " ".join(word['text'] for word in line_words)
                    avg_size = np.mean([word['size'] for word in line_words])
                    
                    # Title characteristics
                    if avg_size > 12 and min_x0 < 100 and max_x1 > first_page.width - 100:
                        title_candidates.append((avg_size, text))
                
                if title_candidates:
                    title_candidates.sort(key=lambda x: x[0], reverse=True)
                    title = title_candidates[0][1]
            
            # Extract headings from all pages
            heading_candidates = []
            for page_number, page in enumerate(pdf.pages, start=1):
                words = page.extract_words(
                    extra_attrs=["size", "fontname"],
                    keep_blank_chars=False
                )
                
                # Group words into lines
                lines = defaultdict(list)
                for word in words:
                    top_key = round(word['top'])
                    lines[top_key].append(word)
                
                # Process each line
                for top in sorted(lines.keys()):
                    line_words = lines[top]
                    if not line_words:
                        continue
                        
                    min_x0 = min(word['x0'] for word in line_words)
                    max_x1 = max(word['x1'] for word in line_words)
                    text = " ".join(word['text'] for word in line_words)
                    max_size = max(word['size'] for word in line_words)
                    
                    # Heading characteristics
                    if (max_size > 9 and 
                        min_x0 > 40 and 
                        max_x1 < page.width - 40 and
                        len(text) < 250):
                        heading_candidates.append({
                            "text": text,
                            "size": max_size,
                            "page": page_number,
                            "y": top
                        })
            
            # Cluster heading levels
            if heading_candidates:
                sizes = [c['size'] for c in heading_candidates]
                if sizes:
                    distinct_sizes = sorted(set(sizes), reverse=True)[:3]
                    
                    # Assign heading levels
                    level_map = {}
                    for i, size in enumerate(distinct_sizes):
                        level_map[size] = f"H{i+1}"
                    
                    # Create outline
                    for candidate in sorted(heading_candidates, key=lambda x: (x['page'], x['y'])):
                        level = level_map.get(candidate['size'], "H1")
                        headings.append({
                            "level": level,
                            "text": candidate['text'],
                            "page": candidate['page']
                        })
    
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
    
    return {
        "title": title,
        "outline": headings
    }

def extract_page_text(pdf_path: str, page_number: int) -> str:
    """Extract text from a specific page"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number <= len(pdf.pages):
                page = pdf.pages[page_number - 1]
                text = page.extract_text()
                return text.strip() if text else ""
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path} page {page_number}: {str(e)}")
        return ""