import requests
from models.model import BaseModel
import openai

from secret import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


class OpenAIModel(BaseModel):
        
    def __init__(self, config):
        super().__init__(config)
    
    def respond(self, question, options):
        instruction = "Given the following multiple-choice question, please respond with just the letter corresponding to the correct answer."
        question = question.split(":")[0]
        choices = "\n".join([f"{chr(65 + i)}) {choice}" for i, choice in enumerate(options)])
        prompt = f"Instructions: {instruction}\n Question: {question}\nOptions: {choices}\nAnswer:"
        completion = openai.Completion.create(
            model=self.config["model"],
            prompt=prompt,
            max_tokens=2,
            temperature=0,
            logprobs=len(options),
        )
        # r = requests.post(url, params=params, headers=headers).json()
        answer = completion.choices[0].text
        try:
            answer = ord(answer.strip()) - ord("A") + 1
        except TypeError:
            print(answer)
            answer = 0
        try:
            response = [1 if i == answer else 0 for i in range(options)]
        except TypeError:
            response = [0] * options
        return response


openai_davinci_base_config = {
    "name": "davinci_2",
    "model": "davinci-002",
}
openai_davinci_base = OpenAIModel(openai_davinci_base_config)