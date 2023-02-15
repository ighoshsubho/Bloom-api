from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

def load_model():
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7", use_cache=True)
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")

def predict(post:dict):
    post_type = post['type_of']
    post_title = post['text']
    prompt = f'This is a {post_type} post on {post_title} \n {post_title} is very important.'
    input_ids = tokenizer(prompt, return_tensors="pt").to(0)
    if post_type == 'blog':
        sample = model.generate(**input_ids, max_length=2000,  top_k=1, temperature=0.9, repetition_penalty = 2.0)
    if post_type == 'article':
        sample = model.generate(**input_ids, max_length=600,  top_k=1, temperature=0.9, repetition_penalty = 2.0)
    if post_type == 'social':
        sample = model.generate(**input_ids, max_length=300,  top_k=1, temperature=0.9, repetition_penalty = 2.0)
    return tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])