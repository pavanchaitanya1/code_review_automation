
from src.util import load_retriever_and_llm, load_test_data, retrieve_similar_docs
from src.util import extract_json_from_text, yes_no, filter_docs
from src.prompts import REVIEW_NEEDED_PROMPT_PATCH, REVIEW_NEEDED_PROMPT_EXAMPLE
from src.prompts import REVIEW_COMMENT_PROMPT_EXAMPLE, REVIEW_COMMENT_PROMPT_PATCH

class CodeReviewer:
    def __init__(self, top_k=5, model_name='mistal'):
        self.top_k = top_k
        self.retriever, self.llm = load_retriever_and_llm(top_k=top_k, model_name=model_name)

    def is_review_needed(self, patch: str):
        prompt = REVIEW_NEEDED_PROMPT_PATCH.format(patch)

        similar_docs = retrieve_similar_docs(self.retriever, patch)
        for i in range(len(similar_docs)):
            doc = similar_docs[i]
            prompt += REVIEW_NEEDED_PROMPT_EXAMPLE.format(i+1, doc.patch, yes_no(doc))
        
        response = self.llm.complete(prompt)
        json_response = extract_json_from_text(response.text)

        if json_response['reviewNeeded'] == 'false':
            return False
        else:
            return True

    def generate_review_comment(self, patch: str):
        self.retriever.similarity_top_k = 20

        prompt = REVIEW_COMMENT_PROMPT_PATCH.format(patch)

        similar_docs = retrieve_similar_docs(self.retriever, patch)
        similar_docs = filter_docs(similar_docs)
        similar_docs = similar_docs[:self.top_k]

        self.retriever.similarity_top_k = self.top_k

        for i in range(len(similar_docs)):
            doc = similar_docs[i]
            prompt += REVIEW_COMMENT_PROMPT_EXAMPLE.format(i+1, doc.patch, yes_no(doc))
        
        response = self.llm.complete(prompt)
        json_response = extract_json_from_text(response.text)

        return json_response['reviewComment']
    


