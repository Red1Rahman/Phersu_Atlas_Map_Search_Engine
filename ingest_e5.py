
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

document_store = ChromaDocumentStore(persist_path=CHROMA_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = Pipeline()
pipeline.add_component("converter", PyPDFToDocument())
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model="intfloat/e5-large-v2"))
pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2))
pipeline.add_component("writer", DocumentWriter(document_store=document_store))

pipeline.connect("converter.documents", "cleaner.documents")
pipeline.connect("cleaner.documents", "splitter.documents")
pipeline.connect("splitter.documents", "embedder.documents")
pipeline.connect("embedder.documents", "writer.documents")

pdf_paths = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

print("Ingesting documents...")
pipeline.run({"converter": {"sources": pdf_paths}})
print("Done.")

end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"Ingestion completed in {elapsed_time:.2f} seconds.")
