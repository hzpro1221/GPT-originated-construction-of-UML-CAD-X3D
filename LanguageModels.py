import torch
import torch.nn as nn

from transformers import pipeline

class TinyLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.pipe = pipeline("text-generation", 
                             model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                             torch_dtype=torch.bfloat16, 
                             device_map="auto")
    
    def generate(self, prompt):
        # Need prompt engeneering for clearer output
        # ---------------------------------------------------
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot" 

            },
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ]
        # ---------------------------------------------------
        prompt = self.pipe.tokenizer.apply_chat_template(messages, 
                                                         tokenize=False, 
                                                         add_generation_prompt=True)
        outputs = self.pipe(prompt, 
                            max_new_tokens=256, 
                            do_sample=True, 
                            temperature=0.7, 
                            top_k=50, 
                            top_p=0.95)
        return outputs[0]["generated_text"]