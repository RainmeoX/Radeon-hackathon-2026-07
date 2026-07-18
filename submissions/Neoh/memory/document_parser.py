import logging
import os
from typing import List, Dict, Any
import pdfplumber
import docx
import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class DocumentParser:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_file(self, file_path: str) -> List[str]:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            if ext == ".pdf":
                return self._parse_pdf(file_path)
            elif ext == ".docx":
                return self._parse_docx(file_path)
            elif ext == ".md":
                return self._parse_markdown(file_path)
            elif ext == ".txt":
                return self._parse_txt(file_path)
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return []
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {str(e)}")
            return []

    def _parse_pdf(self, file_path: str) -> List[str]:
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    texts.append(text.strip())
        return texts

    def _parse_docx(self, file_path: str) -> List[str]:
        doc = docx.Document(file_path)
        texts = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                texts.append(text)
        return texts

    def _parse_markdown(self, file_path: str) -> List[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        md = markdown.Markdown(extensions=["extra"])
        html = md.convert(content)
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        
        return [text.strip()]

    def _parse_txt(self, file_path: str) -> List[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [content.strip()]

    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            if end >= text_length:
                chunk = text[start:]
                if chunk.strip():
                    chunks.append(chunk.strip())
                break

            last_period = text.rfind(". ", start, end)
            last_newline = text.rfind("\n", start, end)
            last_space = text.rfind(" ", start, end)

            split_pos = max(last_period, last_newline, last_space)
            if split_pos > start:
                end = split_pos + 1
            else:
                end = start + self.chunk_size

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start < 0:
                start = 0

        return chunks

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        raw_texts = self.parse_file(file_path)
        all_chunks = []
        
        for text in raw_texts:
            chunks = self.split_text(text)
            all_chunks.extend(chunks)

        documents = []
        for i, chunk in enumerate(all_chunks):
            documents.append({
                "content": chunk,
                "metadata": {
                    "file_name": os.path.basename(file_path),
                    "chunk_index": i,
                    "total_chunks": len(all_chunks),
                },
            })

        return documents