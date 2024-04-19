from weaviate.client import Client
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever

from src.create_index import create_client, create_vector_store, get_storage_context, create_storage_context, create_index

import numpy as np

def main():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.llm = Ollama(model="mistral")

    client = create_client()
    vector_store = create_vector_store(client)
    storage_context = get_storage_context(vector_store)

    #print('index pulling starts')

    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    #print('index pulled')

    test_data = np.load('../data/test_file.npz', allow_pickle=True)['arr_0'][0]['patch']

    print(test_data)

    print('----------------------')

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=25
    )

    similar_docs = retriever.retrieve(test_data)

    for doc in similar_docs:
        print(doc.text)
        print('----------------------')
        print(doc.score)
        print(doc.metadata)
        # print(doc.get_content(metadata_mode='all'))

    # query_engine = index.as_query_engine(similarity_top_k=2)

    # query_string = 'Tell me about ASE project team'
    # response = query_engine.query(query_string)
    # print(response.response)

if __name__ == '__main__':
    main()

