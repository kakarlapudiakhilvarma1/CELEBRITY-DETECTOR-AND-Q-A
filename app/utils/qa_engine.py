import os
import requests


class QAEngine:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    
    def ask_about_celebrity(self, name, question):

        headers = {
            "Authorization" : f"Bearer {self.api_key}",
            "Content-Type" : "application/json"
        }

        prompt = f"""
                    You are an AI Assistant with deep knowledge about celebrities. Your task is to answer questions about {name} concisely and accurately.
                    Always provide factual, up-to-date, and relevant information in a clear manner. Avoid speculation or unnecessary details.

                    Question: {question}

                    """

        payload = {
            "model" : self.model,
            "messages" : [{"role" : "user", "content" : prompt}],
            "max_tokens" : 512,
            "temperature" : 0.5
        }

        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        
        return "Sorry, I couldn't find an answer to your question."