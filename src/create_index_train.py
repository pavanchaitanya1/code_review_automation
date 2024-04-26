from weaviate.client import Client
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import json
import numpy as np

from src.constants import WEAVIATE_INDEX_NAME_TRAIN, PERSIST_DIR_TRAIN

def create_client():
    return Client(url="http://localhost:8080")

def get_unique_patches(data):
    seen = set()
    unique_tuples = []
    for tup in data:
        if tup['patch'] not in seen:
            unique_tuples.append(tup)
            seen.add(tup['patch'])
    return unique_tuples

def create_vector_store(client: Client):
    vector_store = WeaviateVectorStore(
        weaviate_client = client, 
        index_name = WEAVIATE_INDEX_NAME_TRAIN,
    )
    return vector_store

def create_storage_context(vector_store: WeaviateVectorStore):
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    return storage_context

def get_storage_context(vector_store: WeaviateVectorStore):
    storage_context = StorageContext.from_defaults(vector_store = vector_store, persist_dir=PERSIST_DIR_TRAIN)
    return storage_context

def create_index(documents, storage_context: StorageContext):
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

def store_index(index: VectorStoreIndex):
    index.storage_context.persist(persist_dir=PERSIST_DIR_TRAIN)


def main():
    client = create_client()

    data = []
    filenames = ['../data/code_reviewer/diff_quality_estimation/cls-train-chunk-{}.jsonl'.format(i) for i in range(4)]
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    
    data = get_unique_patches(data)

    docs = []

    for i in data:
        docs.append(Document(text=i['patch'], extra_info={'y':i['y'], 'msg':i['msg'] , 'patch_id':i['id']}))

    # docs = docs[:1]

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.llm = Ollama(model="mistral")

    vector_store = create_vector_store(client)
    storage_context = create_storage_context(vector_store)
    
    index = create_index(docs, storage_context)
    store_index(index)

if __name__ == '__main__':
    main()

