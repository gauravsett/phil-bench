from models.model import BaseModel
from anthropic import Anthropic

from secret import ANTHROPIC_API_KEY

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)


class AnthropicModel(BaseModel):
        
    def __init__(self, config):
        super().__init__(config)
    
    def respond(self, question, options):
        instruction = "Given the following multiple-choice question, please respond with just the letter corresponding to the correct answer."
        question = question.split(":")[0]
        options = "\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])
        prompt = f"Instructions: {instruction}\n Question: {question}\nOptions: {options}\nAnswer: "
        completion = anthropic.completions.create(
            model=self.config["model"],
            prompt=prompt,
            max_tokens_to_sample=2,
            temperature=0,
        )
        answer = ord(completion.completion.split("Answer: ")[1].strip()) - ord("A") + 1
        response = [1 if i == answer else 0 for i in range(options)]
        return response


claude2_base_config = {
    "model": "claude-2",
    "api_key": ANTHROPIC_API_KEY,
}
claude2_base = AnthropicModel(claude2_base_config)