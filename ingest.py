from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
import os
from datetime import datetime

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
split_length = 75  # Number of sentences/paragraphs/words to split by
timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")

ds = ChromaDocumentStore(persist_path=CHHROMA_SAVEPATH)
pipe = Pipeline()
pipe.add_component("converter", PyPDFToDocument())
pipe.add_component("cleaner", DocumentCleaner())
pipe.add_component("splitter", DocumentSplitter(split_by = split_by, split_length = split_length))
pipe.add_component("writer", DocumentWriter(document_store = ds))

pipe.connect("converter", "cleaner")
pipe.connect("cleaner", "splitter")
pipe.connect("splitter", "writer")

try:
    initial_doc_count = ds.count_documents()
    print(f"Initial document count in store: {initial_doc_count}")
    if initial_doc_count == 0 or input("Document store is not empty. Re-index? (y/n): ").lower() == 'y':
        pipe.run({"converter": {"sources": file_names}})
        print("Pipeline finished running. Documents have been written to the ChromaDocumentStore.")
    else:
        print("Skipping re-indexing. Using existing documents in the store.")

    retrieved_documents = ds.filter_documents()

    if retrieved_documents:
        print(f"\n--- Found {len(retrieved_documents)} documents in the store ---")
        file_name = f"./data/output/test_run_{split_by}_{split_length}_{timestamp_str}.txt"
        with open(file_name, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(retrieved_documents):
                f.write(f"\nDocument {i+1} (ID: {doc.id}):")
                f.write("Content:")
                f.write(doc.content)
                f.write("Metadata:")
                f.write(str(doc.meta))
                f.write("-" * 50)
    else:
        print("\nNo documents found in the ChromaDocumentStore. Check pipeline configuration or PDF content.")

except Exception as e:
    print(f"An error occurred during pipeline execution: {e}")
    print("Please ensure the PDF path is correct and the file is readable.")