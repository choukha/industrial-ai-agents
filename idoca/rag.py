"""Retrieval-Augmented Generation (RAG) system for IDOCA."""

import logging
import traceback
from typing import List, Dict, Any, Optional, Literal

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores.milvus import Milvus
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from idoca.config import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, DEFAULT_VECTOR_DB_TYPE,
    DEFAULT_COLLECTION_NAME
)
from idoca.utils import MockEmbeddings, MockChatModel

logger = logging.getLogger("idoca.rag")

# Vector DB type definition
VectorDBType = Literal["chroma", "faiss", "milvus"]

class RAGSystem:
    """Manages the RAG pipeline: embeddings, vector store, and retrieval chain."""
    
    def __init__(self, 
                 embedding_model_name: str = DEFAULT_EMBEDDING_MODEL, 
                 llm_model_name: str = DEFAULT_LLM_MODEL, 
                 collection_name: str = DEFAULT_COLLECTION_NAME, 
                 vector_db_type: VectorDBType = DEFAULT_VECTOR_DB_TYPE,
                 vector_db_config: Optional[Dict[str, Any]] = None):
        
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.collection_name = collection_name
        self.vector_db_type = vector_db_type
        self.vector_db_config = vector_db_config or {}
        self.embeddings: Optional[Embeddings] = None
        self.llm = None
        self.vector_store = None
        self.retriever: Optional[VectorStoreRetriever] = None
        self.rag_chain = None
        self.processed_docs: List[Document] = []
        self.status_messages: List[str] = []
        self._initialize_models()

    def _initialize_models(self):
        """Initialize embedding and LLM models, with fallback to mock implementations."""
        # Initialize embedding model
        try:
            self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)
            self.status_messages.append(f"✅ Emb:'{self.embedding_model_name}' OK.")
            logger.info(f"Emb '{self.embedding_model_name}' loaded.")
        except Exception as e:
            self.embeddings = MockEmbeddings()
            self.status_messages.append(f"⚠️ Emb:MOCK ('{self.embedding_model_name}' fail:{e}).")
            logger.warning(f"Emb MOCK.")
        
        # Initialize LLM
        try:
            self.llm = ChatOllama(model=self.llm_model_name)
            self.status_messages.append(f"✅ LLM:'{self.llm_model_name}' OK.")
            logger.info(f"LLM '{self.llm_model_name}' loaded.")
        except Exception as e:
            self.llm = MockChatModel()
            self.status_messages.append(f"⚠️ LLM:MOCK ('{self.llm_model_name}' fail:{e}).")
            logger.warning(f"LLM MOCK.")

    def add_documents(self, documents: List[Document]):
        """Add documents to the internal document store."""
        if documents:
            valid_docs = [d for d in documents if d and d.page_content]
            self.processed_docs.extend(valid_docs)
            logger.info(f"Added {len(valid_docs)} docs. Total:{len(self.processed_docs)}")
            if len(valid_docs) < len(documents):
                logger.warning(f"Skipped {len(documents) - len(valid_docs)} empty docs.")
        else:
            logger.warning("No docs to add.")

    def clear_documents(self):
        """Clear all documents and reset the RAG pipeline."""
        self.processed_docs = []
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None
        self.status_messages.append("ℹ️ Data cleared, RAG reset.")
        logger.info("Data cleared.")

    def build_vector_store(self, force_rebuild: bool = False) -> bool:
        """Build or rebuild the vector store from processed documents."""
        if not self.processed_docs:
            self.status_messages.append("⚠️ VS:No docs.")
            logger.warning("VS:No docs.")
            return False
            
        if self.vector_store and not force_rebuild:
            logger.info("VS:Exists.")
            return True
            
        if not self.embeddings:
            self.status_messages.append("❌ VS:Emb N/A.")
            logger.error("VS:Emb N/A.")
            return False
            
        logger.info(f"Building VS '{self.collection_name}' with {len(self.processed_docs)} docs...")
        
        try:
            if self.vector_store and force_rebuild:
                self.vector_store = None
                self.retriever = None
            
            # Build vector store based on specified type
            if self.vector_db_type == "chroma":
                return self._build_chroma_vector_store()
            elif self.vector_db_type == "faiss":
                return self._build_faiss_vector_store()
            elif self.vector_db_type == "milvus":
                return self._build_milvus_vector_store()
            else:
                self.status_messages.append(f"❌ VS:Unknown vector DB type '{self.vector_db_type}'")
                logger.error(f"VS:Unknown vector DB type '{self.vector_db_type}'")
                return False
                
        except Exception as e:
            self.status_messages.append(f"❌ VS:Build fail-{e}")
            logger.error(f"VS Build fail:{e}\n{traceback.format_exc()}")
            self.vector_store = None
            self.retriever = None
            return False
    
    def _build_chroma_vector_store(self) -> bool:
        """Build a Chroma vector store."""
        try:
            # Create a custom filter function to handle Docling metadata
            def filter_docling_metadata(metadata_dict):
                """Filter complex nested metadata from Docling documents."""
                if not metadata_dict:
                    return {}
                    
                filtered = {}
                
                # Copy simple metadata fields
                for key, value in metadata_dict.items():
                    # Keep only simple data types (strings, numbers, booleans)
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        filtered[key] = value
                    elif key == "headings" and isinstance(value, list):
                        # Convert headings list to a string
                        filtered[key] = " > ".join(str(h) for h in value if h)
                
                return filtered
            
            # Process documents to filter complex metadata
            docs_with_filtered_metadata = []
            for doc in self.processed_docs:
                # Create a new document with filtered metadata
                filtered_metadata = filter_docling_metadata(doc.metadata)
                
                # Ensure source is preserved
                if "source" not in filtered_metadata and hasattr(doc.metadata, "get"):
                    source = doc.metadata.get("source", "unknown")
                    if isinstance(source, (str, int, float, bool)) or source is None:
                        filtered_metadata["source"] = source
                
                filtered_doc = Document(
                    page_content=doc.page_content,
                    metadata=filtered_metadata
                )
                docs_with_filtered_metadata.append(filtered_doc)
                
            # Extract Chroma-specific config 
            persist_directory = self.vector_db_config.get("persist_directory", None)
            
            self.vector_store = Chroma.from_documents(
                documents=docs_with_filtered_metadata,  # Use filtered documents
                embedding=self.embeddings, 
                collection_name=self.collection_name,
                persist_directory=persist_directory
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            self.status_messages.append(f"✅ VS:ChromaDB Built({len(self.processed_docs)} items).")
            logger.info("ChromaDB Built.")
            return True
        except Exception as e:
            self.status_messages.append(f"❌ VS:ChromaDB Build fail-{e}")
            logger.error(f"ChromaDB Build fail:{e}\n{traceback.format_exc()}")
            return False
    
    def _build_faiss_vector_store(self) -> bool:
        """Build a FAISS vector store."""
        try:
            # Extract FAISS-specific config
            index_name = self.vector_db_config.get("index_name", "industrial_faiss")
            
            self.vector_store = FAISS.from_documents(
                documents=self.processed_docs,
                embedding=self.embeddings,
            )
            
            # Save index if path is provided
            if "save_path" in self.vector_db_config:
                self.vector_store.save_local(self.vector_db_config["save_path"], index_name)
                
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            self.status_messages.append(f"✅ VS:FAISS Built({len(self.processed_docs)} items).")
            logger.info("FAISS Built.")
            return True
        except Exception as e:
            self.status_messages.append(f"❌ VS:FAISS Build fail-{e}")
            logger.error(f"FAISS Build fail:{e}\n{traceback.format_exc()}")
            return False
    
    def _build_milvus_vector_store(self) -> bool:
        """Build a Milvus vector store."""
        try:
            # Extract Milvus-specific config with defaults
            connection_args = self.vector_db_config.get("connection_args", {
                "host": "localhost", 
                "port": "19530"
            })
            
            # Additional Milvus parameters
            collection_name = self.vector_db_config.get("collection_name", self.collection_name)
            drop_old = self.vector_db_config.get("drop_old", False)
            
            self.vector_store = Milvus.from_documents(
                documents=self.processed_docs,
                embedding=self.embeddings,
                collection_name=collection_name,
                connection_args=connection_args,
                drop_old=drop_old
            )
            
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            self.status_messages.append(f"✅ VS:Milvus Built({len(self.processed_docs)} items).")
            logger.info("Milvus Built.")
            return True
        except Exception as e:
            self.status_messages.append(f"❌ VS:Milvus Build fail-{e}")
            logger.error(f"Milvus Build fail:{e}\n{traceback.format_exc()}")
            return False

    def initialize_rag_chain(self) -> bool:
        """Initialize the RAG chain with the retriever and LLM."""
        if self.rag_chain:
            logger.info("RAG Chain:Exists.")
            return True
            
        if not self.retriever:
            self.status_messages.append("⚠️ RAG Chain:Retriever N/A.")
            logger.warning("RAG Chain:No retriever.")
            return False
            
        if not self.llm:
            self.status_messages.append("⚠️ RAG Chain:LLM N/A.")
            logger.warning("RAG Chain:No LLM.")
            return False
            
        logger.info("Initializing RAG chain...")
        
        try:
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm, 
                chain_type="stuff", 
                retriever=self.retriever, 
                return_source_documents=True
            )
            self.status_messages.append("✅ RAG Chain:Initialized.")
            logger.info("RAG Chain Initialized.")
            return True
            
        except Exception as e:
            self.status_messages.append(f"❌ RAG Chain:Init fail-{e}")
            logger.error(f"RAG Chain Init fail:{e}\n{traceback.format_exc()}")
            self.rag_chain = None
            return False

    def query(self, query_text: str) -> Optional[Dict[str, Any]]:
        """Query the RAG system with user input."""
        if not self.rag_chain:
            logger.error("RAG chain N/A.")
            return {"error": "RAG chain N/A."}
            
        logger.info(f"RAG query:'{query_text[:70]}...'")
        
        try:
            result = self.rag_chain.invoke({"query": query_text})
            if isinstance(result, dict) and "query" in result and "result" in result:
                logger.info("RAG query OK.")
                return result
            else:
                logger.warning(f"RAG query bad format:{type(result)}")
                return {"error": "RAG query bad format.", "raw_result": result}
                
        except Exception as e:
            logger.error(f"RAG query fail:{e}\n{traceback.format_exc()}")
            return {"error": f"RAG query fail:{str(e)}"}

    def get_status(self, concise=False) -> List[str]:
        """Get the current status of the RAG system components."""
        e_ok = isinstance(self.embeddings, OllamaEmbeddings)
        l_ok = isinstance(self.llm, ChatOllama)
        v_ok = bool(self.vector_store)
        c_ok = bool(self.rag_chain)
        
        vector_db_name = self.vector_db_type.upper()
        
        s = (f"RAG:{'OK' if e_ok and l_ok and v_ok and c_ok else 'Needs Attention'} "
             f"(Docs:{len(self.processed_docs)},Emb:{'OK' if e_ok else 'F'},"
             f"LLM:{'OK' if l_ok else 'F'},{vector_db_name}:{'OK' if v_ok else 'F'},Chain:{'OK' if c_ok else 'F'})")
             
        if concise:
            return [s]
            
        es = f"Ollama '{self.embedding_model_name}'" if e_ok else ("Mock" if isinstance(self.embeddings, MockEmbeddings) else "NL")
        ls = f"Ollama '{self.llm_model_name}'" if l_ok else ("Mock" if isinstance(self.llm, MockChatModel) else "NL")
        
        return [
            f"--- RAG Status ---", 
            f"Emb:{es}", 
            f"LLM:{ls}",
            f"Docs:{len(self.processed_docs)}", 
            f"VS:{vector_db_name}{'(Built)' if v_ok else '(NB)'}",
            f"Chain:{'Init' if c_ok else 'NI'}"
        ] + self.status_messages[-1:]