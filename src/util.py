from weaviate.client import Client
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from src.constants import PERSIST_DIR, WEAVIATE_INDEX_NAME
from src.patch import Data
from src.llm import LLM, GPTModel, MistralModel, PerplixityModel

import numpy as np

import re
import json

class LazyDecoder(json.JSONDecoder):
    def decode(self, s, **kwargs):
        regex_replacements = [
            (re.compile(r'([^\\])\\([^\\])'), r'\1\\\\\2'),
            (re.compile(r',(\s*])'), r'\1'),
        ]
        for regex, replacement in regex_replacements:
            s = regex.sub(replacement, s)
        return super().decode(s, **kwargs)

def load_retriever_and_llm(top_k=5, model_name='mistral', use_ollama=False):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    client = Client(url="http://localhost:8080")
    vector_store = WeaviateVectorStore(client, index_name = WEAVIATE_INDEX_NAME)
    storage_context = StorageContext.from_defaults(vector_store = vector_store, persist_dir=PERSIST_DIR)

    # print('index pulling starts\n')

    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    # print('index pulled successfully\n')

    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    if model_name == 'GPT':
        llm = GPTModel()
    elif model_name == 'mistral':
        llm = MistralModel()
    elif model_name == 'perplexity':
        llm = PerplixityModel()
    elif use_ollama:
        llm = LLM(model_name)
    else:
        llm = Ollama(model=model_name, request_timeout=120)
    return retriever, llm

def load_test_data():
    test_data_file = '../data/test_file.npz'
    test_data = np.load(test_data_file, allow_pickle=True)['arr_0']
    return [Data(i) for i in test_data]

def retrieve_similar_docs(retriever: VectorIndexRetriever, patch: str):
    similar_patches = retriever.retrieve(patch)
    docs = []
    for doc in similar_patches:
        patch_object = {}
        patch_object['patch'] = doc.text
        patch_object['score'] = doc.score
        metadata = doc.metadata
        patch_object['proj'] = metadata['proj']
        patch_object['y'] = metadata['y']
        patch_object['msg'] = metadata['msg']
        docs.append(Data(patch_object))
    return docs

def extract_json_from_text(text: str):
    try:
        text = text.replace('\n', "")
        json_pattern = r'{.*?}'
        json_strings = re.findall(json_pattern, text)
        if len(json_strings) == 0:
            print('No Proper Json in the LLM Response: \n' + text)
            return
        elif len(json_strings) > 1:
            print('Multiple Proper Json in the LLM Response: \n' + text)
            return
        else:
            json_string = json_strings[0]
            json_data = json.loads(json_string, cls=LazyDecoder)
            return json_data
    except Exception as ex:
        print(text)
        raise ex
    
def yes_no(doc):
    if doc.y == 1:
        return 'true'
    else:
        return 'false'
    
def filter_docs(docs):
    return_docs = []
    for doc in docs:
        if doc.y == 1:
            return_docs.append(doc)
    return return_docs

