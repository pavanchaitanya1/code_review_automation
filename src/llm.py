import ollama
from openai import OpenAI

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