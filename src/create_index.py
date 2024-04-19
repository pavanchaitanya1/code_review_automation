from weaviate.client import Client
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import json
import numpy as np

from src.constants import WEAVIATE_INDEX_NAME, PERSIST_DIR

def create_client():
    client = Client(
        url="http://localhost:8080"
    )
    return client

def store_test_data(test_data):
    test_file_name = '../data/test_file.npz'
    np.savez(test_file_name, test_data)


def load_docs(filename: str):
    docs = []
    
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    projects = []
    for data1 in data:
        projects.append(data1['proj'])
    projects = set(projects)

    project_data = {}
    for project in projects:
        project_data[project] = []
    for data1 in data:
        project_data[data1['proj']].append(data1)

    train_data = []

    test_data = []

    for project in project_data:
        t_list = project_data[project]
        if len(t_list) > 4:
            test_data.extend(t_list[:2])
            train_data1 = t_list[2:]
        else:
            train_data1 = t_list
        for i in train_data1:
            train_data.append(Document(text=i['patch'], extra_info={'y':i['y'], 'msg':i['msg'], 'proj':i['proj']}))
    
    store_test_data(test_data)

    return train_data

def create_vector_store(client: Client):
    vector_store = WeaviateVectorStore(
        weaviate_client = client, 
        index_name = WEAVIATE_INDEX_NAME,
    )
    return vector_store

def create_storage_context(vector_store: WeaviateVectorStore):
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    return storage_context

def get_storage_context(vector_store: WeaviateVectorStore):
    storage_context = StorageContext.from_defaults(vector_store = vector_store, persist_dir=PERSIST_DIR)
    return storage_context

def create_index(documents, storage_context: StorageContext):
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

def store_index(index: VectorStoreIndex):
    index.storage_context.persist(persist_dir=PERSIST_DIR)


def main():
    client = create_client()
    filename = '../data/code_reviewer/diff_quality_estimation/cls-test.jsonl'
    docs = load_docs(filename)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.llm = Ollama(model="mistral")

    vector_store = create_vector_store(client)
    storage_context = create_storage_context(vector_store)
    
    index = create_index(docs, storage_context)
    store_index(index)

if __name__ == '__main__':
    main()

