
from src.util import load_retriever_and_llm, load_test_data, retrieve_similar_docs
from src.util import extract_json_from_text, yes_no, filter_docs
from src.prompts import *

class CodeReviewer:
    def __init__(self, top_k=1, model_name='mistral', use_ollama=False, use_rag=True, use_yes_no=True, github_bot=False, use_train_store=False):
        self.top_k = top_k
        self.use_ollama = use_ollama
        self.model_name = model_name
        self.use_rag = use_rag
        self.use_yes_no = use_yes_no
        self.github_bot = github_bot
        # print('github bot', self.github_bot)
        self.retriever, self.llm = load_retriever_and_llm(top_k=top_k, model_name=model_name, use_ollama=use_ollama, use_train_store=use_train_store)

    def is_review_needed(self, patch: str):
        prompt = ''

        if self.use_rag:
            similar_docs = retrieve_similar_docs(self.retriever, patch)

            # print('Simiar Docs Length: ', len(similar_docs))

            for i in range(len(similar_docs)):
                doc = similar_docs[i]
                prompt += REVIEW_NEEDED_PROMPT_EXAMPLE.format(i+1, doc.patch, yes_no(doc))
            
            prompt += REVIEW_NEEDED_PROMPT_PATCH.format(patch)
        else:
            prompt += REVIEW_NEEDED_PROMPT_NO_RAG.format(patch)

        # print("--------")
        # print(prompt)
        # print("--------")
        
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
        prompt = ''

        if self.use_rag:
            if self.use_yes_no:
                self.retriever.similarity_top_k = 20
            similar_docs = retrieve_similar_docs(self.retriever, patch)

            if self.use_yes_no:
                similar_docs = filter_docs(similar_docs)
                similar_docs = similar_docs[:self.top_k]
                self.retriever.similarity_top_k = self.top_k

            # print('Simiar Docs Length: ', len(similar_docs))

            for i in range(len(similar_docs)):
                doc = similar_docs[i]
                prompt += REVIEW_COMMENT_PROMPT_EXAMPLE.format(i+1, doc.patch, doc.msg)
            
            if self.use_yes_no:
                if self.github_bot:
                    prompt+= REVIEW_COMMENT_PROMPT_GITHUB.format(patch)
                else:
                    prompt += REVIEW_COMMENT_PROMPT_PATCH.format(patch)
            else:
                prompt += REVIEW_COMMENT_PROMPT_NO_YES_NO.format(patch)

        else:
            prompt += REVIEW_COMMENT_PROMPT_NO_RAG.format(patch)

        # print("--------")
        # print(prompt)
        # print("--------")
        
        response = self.llm.complete(prompt)
        json_response = extract_json_from_text(response)

        if self.github_bot:
            # print(json_response['reviewComment'])
            # print(json_response['lineNumber'])
            return json_response['reviewComment'], json_response['lineNumber']
        
        return json_response['reviewComment']
    


