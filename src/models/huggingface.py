from models.model import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class HuggingFaceModel(BaseModel):
        
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"],
            return_dict=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model"],
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict=True
        ).to(device)
    
    def respond(self, question, options):
        instruction = "Given the following multiple-choice question, please respond with just the letter corresponding to the correct answer."
        question = question.split(":")[0]
        choices = "\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])
        prompt = f"Instructions: {instruction}\n Question: {question}\nOptions: {choices}\nAnswer: "
        response = []
        for i in range(len(options)):
            option = chr(65 + i)
            with torch.no_grad():
                input_ids = self.tokenizer(prompt + option, return_tensors="pt").input_ids.to(device)
                logit = self.model(input_ids, labels=input_ids).logits.item()
                response += [logit]
        response = torch.softmax(torch.tensor(response), dim=0).tolist()
        return response

