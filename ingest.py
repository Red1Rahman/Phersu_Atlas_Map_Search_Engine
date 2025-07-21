from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.converters import PDFMinerToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder # Crucial for embeddings
import os
from datetime import datetime
import time
import logging

# Set logging level to INFO for normal operation
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
start = time.time()


def get_file_names_in_folder(folder_path):
    """
    Returns a list of all file names (not including directories) in the specified folder.
    """
    file_names = []
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)
        if os.path.isfile(item_path):
            file_names.append(item_path)
    return file_names

file_names = get_file_names_in_folder("./data/pdfs/")
CHHROMA_SAVEPATH = "./data/chroma_db"
split_by = "sentence"  # Options: "sentence", "paragraph", "word"
split_length = 10
timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")

# Initialize ChromaDocumentStore, ensuring cosine distance is set
ds = ChromaDocumentStore(persist_path=CHHROMA_SAVEPATH, distance_function='cosine')

pipe = Pipeline()
pipe.add_component("converter", PDFMinerToDocument())
pipe.add_component("cleaner", DocumentCleaner())
pipe.add_component("splitter", DocumentSplitter(split_by=split_by, split_length=split_length))
pipe.add_component("embedder", SentenceTransformersDocumentEmbedder(model="all-MiniLM-L6-v2")) # Add the document embedder
pipe.add_component("writer", DocumentWriter(document_store=ds))

pipe.connect("converter", "cleaner")
pipe.connect("cleaner", "splitter")
pipe.connect("splitter", "embedder") # Connect splitter to embedder
pipe.connect("embedder", "writer")   # Connect embedder to writer


try:
    initial_doc_count = ds.count_documents()
    logging.info(f"Initial document count in store: {initial_doc_count}")
    
    # Always re-index for a clean start after reset
    logging.info("Running ingestion pipeline to re-index documents...")
    pipe.run({"converter": {"sources": file_names}})
    logging.info("Pipeline finished running. Documents have been written to the ChromaDocumentStore.")

    retrieved_documents = ds.filter_documents()

    if retrieved_documents:
        logging.info(f"\n--- Found {len(retrieved_documents)} documents in the store ---")
        file_name = f"./data/output/test_run_{split_by}_{split_length}_{timestamp_str}.txt"
        
        output_dir = os.path.dirname(file_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(file_name, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(retrieved_documents):
                f.write(f"\nDocument {i+1} (ID: {doc.id}):\n")
                f.write("Content:\n")
                f.write(doc.content + "\n")
                f.write("Metadata:\n")
                f.write(str(doc.meta) + "\n")
                f.write("-" * 50 + "\n")
        logging.info(f"Indexed document details written to: {file_name}")
    else:
        logging.info("\nNo documents found in the ChromaDocumentStore. Check pipeline configuration or PDF content.")

except Exception as e:
    logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
    logging.error("Please ensure the PDF path is correct and the file is readable.")

end = time.time()
logging.info(f"Time taken: {end - start} seconds")
