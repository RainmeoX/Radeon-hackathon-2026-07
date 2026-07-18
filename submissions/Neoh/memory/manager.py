import logging
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from .document_parser import DocumentParser

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(
        self,
        index_path: str = "./data/faiss_index",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        self.vector_store = VectorStore(
            index_path=index_path,
            embedding_model=embedding_model,
        )
        self.document_parser = DocumentParser(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.top_k = top_k
        self.short_term_memory: List[Dict[str, str]] = []

    def add_document(self, file_path: str) -> int:
        logger.info(f"Processing document: {file_path}")
        documents = self.document_parser.process_file(file_path)
        
        if documents:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} chunks from {file_path}")
            return len(documents)
        return 0

    def add_documents(self, file_paths: List[str]) -> int:
        total_added = 0
        for file_path in file_paths:
            added = self.add_document(file_path)
            total_added += added
        return total_added

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k if top_k is not None else self.top_k
        results = self.vector_store.search(query, top_k=k)
        
        context = []
        for result in results:
            context.append({
                "content": result["content"],
                "metadata": result["metadata"],
            })
        
        return context

    def get_context(self, query: str, top_k: Optional[int] = None) -> str:
        results = self.search(query, top_k=top_k)
        
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[参考文档 {i+1}]\n{result['content']}\n")
        
        return "\n".join(context_parts)

    def add_short_term_memory(self, message: Dict[str, str]):
        self.short_term_memory.append(message)
        
        if len(self.short_term_memory) > 20:
            self.short_term_memory = self.short_term_memory[-20:]

    def get_short_term_memory(self) -> List[Dict[str, str]]:
        return self.short_term_memory.copy()

    def clear_short_term_memory(self):
        self.short_term_memory = []

    def get_document_count(self) -> int:
        return self.vector_store.get_document_count()

    def clear_long_term_memory(self):
        self.vector_store.clear()

    def reset(self):
        self.clear_short_term_memory()
        self.clear_long_term_memory()