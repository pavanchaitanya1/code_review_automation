
from src.util import load_retriever_and_llm, load_test_data, retrieve_similar_docs
from src.util import extract_json_from_text, yes_no, filter_docs
from src.prompts import REVIEW_NEEDED_PROMPT_PATCH, REVIEW_NEEDED_PROMPT_EXAMPLE
from src.prompts import REVIEW_COMMENT_PROMPT_EXAMPLE, REVIEW_COMMENT_PROMPT_PATCH

class CodeReviewer:
    def __init__(self, top_k=5, model_name='mistral', use_ollama=False):
        self.top_k = top_k
        self.use_ollama = use_ollama
        self.model_name = model_name
        self.retriever, self.llm = load_retriever_and_llm(top_k=top_k, model_name=model_name, use_ollama=use_ollama)

    def is_review_needed(self, patch: str):
        prompt = ''

        similar_docs = retrieve_similar_docs(self.retriever, patch)

        print('Simiar Docs Length: ', len(similar_docs))

        for i in range(len(similar_docs)):
            doc = similar_docs[i]
            prompt += REVIEW_NEEDED_PROMPT_EXAMPLE.format(i+1, doc.patch, yes_no(doc))
        
        prompt += REVIEW_NEEDED_PROMPT_PATCH.format(patch)
        
        response = self.llm.complete(prompt)
        json_response = extract_json_from_text(response)

        try:
            if json_response['reviewNeeded'] == 'false':
                return False
            else:
                return True
        except Exception as ex:
            print('Error in Json: \n' + response)
            raise Exception('Json Parsing Failed')


    def generate_review_comment(self, patch: str):
        self.retriever.similarity_top_k = 20

        prompt = ''

        similar_docs = retrieve_similar_docs(self.retriever, patch)
        similar_docs = filter_docs(similar_docs)
        similar_docs = similar_docs[:self.top_k]

        self.retriever.similarity_top_k = self.top_k

        print('Simiar Docs Length: ', len(similar_docs))

        for i in range(len(similar_docs)):
            doc = similar_docs[i]
            prompt += REVIEW_COMMENT_PROMPT_EXAMPLE.format(i+1, doc.patch, doc.msg)
        
        prompt += REVIEW_COMMENT_PROMPT_PATCH.format(patch)
        
        response = self.llm.complete(prompt)
        json_response = extract_json_from_text(response)

        return json_response['reviewComment']
    


