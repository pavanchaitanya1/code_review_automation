import ollama
from openai import OpenAI
import os

class LLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def complete(self, prompt):
        response = ollama.chat(model=self.model_name, messages=[
            {
                'role' : 'user',
                'content' : prompt
            }
        ])
        return response['message']['content']
    
class GPTModel:
    def __init__(self):
        self.client = OpenAI()

    def complete(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a lead software engineer performing code reviews."},
                {"role": "user", "content": prompt}
            ]
            )
        return response.choices[0].message.content
    
class MistralModel:
    def __init__(self):
        api_key = os.environ.get('MISTRAL_API_KEY')
        self.client = OpenAI(
            api_key =  os.environ.get('MISTRAL_API_KEY'),
            base_url="https://api.lemonfox.ai/v1",
        )

    def complete(self, prompt):
        response = self.client.chat.completions.create(
        messages=[
            { "role": "system", "content": "You are a lead software engineer performing code reviews." },
            { "role": "user", "content": prompt }
        ],
        model="mixtral-chat"
        )

        return response.choices[0].message.content
    
class PerplixityModel:
    def __init__(self):
        api_key = os.environ.get('PERPLEXITY_API_KEY')
        self.client = OpenAI(api_key = api_key, base_url = 'https://api.perplexity.ai')
    
    def complete(self, prompt):
        response = self.client.chat.completions.create(
             messages=[
            { "role": "system", "content": "You are a lead software engineer performing code reviews." },
            { "role": "user", "content": prompt }
        ],
        model="llama-3-70b-instruct"
        )
        return response.choices[0].message.content