from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.converters import PDFMinerToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
import os
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
start = time.time()

def get_file_names_in_folder(folder_path):
    return [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, name)) and name.lower().endswith(".pdf")
    ]

file_names = get_file_names_in_folder("./data/pdfs/")
CHHROMA_SAVEPATH = "./data/chroma_db"
split_by = "sentence"
split_length = 10
timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")

chroma_store = ChromaDocumentStore(persist_path=CHHROMA_SAVEPATH, distance_function='cosine')

pipe = Pipeline()
pipe.add_component("converter", PDFMinerToDocument())
pipe.add_component("cleaner", DocumentCleaner())
pipe.add_component("splitter", DocumentSplitter(split_by=split_by, split_length=split_length))
pipe.add_component("embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-mpnet-base-v2"))
pipe.add_component("writer", DocumentWriter(document_store=chroma_store))

pipe.connect("converter", "cleaner")
pipe.connect("cleaner", "splitter")
pipe.connect("splitter", "embedder")
pipe.connect("embedder", "writer")

try:
    logging.info(f"Initial document count: {chroma_store.count_documents()}")
    pipe.run({"converter": {"sources": file_names}})
    logging.info(f"Updated document count: {chroma_store.count_documents()}")
except Exception as e:
    logging.error(f"Pipeline error: {e}", exc_info=True)

end = time.time()
logging.info(f"Time taken: {end - start:.2f} seconds")
