import ollama

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