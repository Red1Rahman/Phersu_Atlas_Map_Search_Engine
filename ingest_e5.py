from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
import os
import time
import logging

start_time = time.time()
CHROMA_PATH = "./data/chroma_db_e5_embeddings"
PDF_DIR = "./data/pdfs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

document_store = ChromaDocumentStore(persist_path=CHROMA_PATH)

pipeline = Pipeline()
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2))
pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model="intfloat/e5-large-v2"))
pipeline.add_component("writer", DocumentWriter(document_store=document_store))

pipeline.connect("cleaner.documents", "splitter.documents")
pipeline.connect("splitter.documents", "embedder.documents")
pipeline.connect("embedder.documents", "writer.documents")

pdf_paths = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

print("Ingesting documents...")
converter = PyPDFToDocument()

for path in pdf_paths:
    logger.info(f"Processing {path}...")
    raw_docs = converter.run(sources=[path])["documents"]

    for doc in raw_docs:
        # Ensure meta field is initialized and includes filename
        if not hasattr(doc, "meta") or doc.meta is None:
            doc.meta = {}
        doc.meta["file_path"] = os.path.basename(path)

    # Pass through the rest of the pipeline
    result = pipeline.run({"cleaner": {"documents": raw_docs}})


end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"Ingestion completed in {elapsed_time:.2f} seconds.")
