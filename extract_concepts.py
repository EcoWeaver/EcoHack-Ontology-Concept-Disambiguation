import os.path
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
import json
from tqdm import tqdm
import torch
from transformers import pipeline
from load_data import load_abstracts
import re
from nltk.stem import PorterStemmer

persona = "You are an expert in ecological ontologies, specifically the ENVO ontology. Your task is to extract ecological concepts/terms that match the ENVO ontology from text. These concepts can be single-word or multi-word phrases that describe environmental or ecological entities, processes, or characteristics."

def create_prompt(text):
    return f"""
    
    Here are some examples:
    
    Text: 'The mangrove forest ecosystem provides critical habitat for many species.'
    Concepts: ['mangrove forest ecosystem', 'habitat']
    
    Text: 'Marine sediment contains important biogeochemical markers.'
    Concepts: ['marine sediment', 'biogeochemical markers']
    
    Text: 'Freshwater wetlands are essential for biodiversity and water filtration.'
    Concepts: ['freshwater wetlands', 'biodiversity', 'water filtration']
    
    Text: 'The soil organic carbon content is influenced by climate and vegetation.'
    Concepts: ['soil organic carbon content', 'climate', 'vegetation']
    
    Now, analyze the following text and extract all relevant ecological concepts:
    
    Text: {text}
    
    Format your response in the following way, and do not use any additional introductory sentence:
    CONCEPTS: [Comma separated list of concepts]
    END.
    """

def generate_continuations(text, pipe):
    with torch.no_grad():
        messages = [{"role": "system", "content": persona}, {"role": "user", "content": create_prompt(text)}]

        response = pipe(messages, num_return_sequences=1, temperature=1.2, pad_token_id=pipe.tokenizer.eos_token_id)
        response = response[0]["generated_text"][2]["content"]

        start_idx = response.find("CONCEPTS:") + len("CONCEPTS:")
        end_idx = response.find("END.")
        if start_idx < 0 or end_idx < 0:
            return None

        concepts = [x.strip() for x in response[start_idx:end_idx].strip().split(",")]
        return concepts

def load_model():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipe = pipeline("text-generation", model=model_name, device_map="auto", torch_dtype=torch.bfloat16)
    return pipe

def generate_concepts():
    abstracts = load_abstracts()

    pipe =  load_model()

    idx = 0
    bar = tqdm(total=len(abstracts), initial=idx, desc="Generating concepts...", smoothing=0.5)
    for paper_idx in abstracts:
        idx += 1

        if os.path.exists(f"detected_concepts/{paper_idx}.json"):
            continue

        title = abstracts[paper_idx]['title']
        abstract = abstracts[paper_idx]['abstract']
        if title[-1].isalpha():
            title += "."
        text = f"{title} {abstract}"

        continuations = generate_continuations(text, pipe)

        if continuations is None:
            continue

        with open(f"detected_concepts/{paper_idx}.json", "w") as f:
            json.dump(continuations, f)
        bar.update(1)



if __name__ == '__main__':
    generate_concepts()