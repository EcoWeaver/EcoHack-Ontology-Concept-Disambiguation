import os.path

if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
import json
from tqdm import tqdm
import torch
from transformers import pipeline
from load_data import load_abstracts, load_concepts


def create_prompt(text, word):
    return f"""This is a scientific abstract:
            
            {text}

            Task: Generate a definition for "{word}" that contains its meaning as it is used in the abstract.
            Please do not use the word itself in its definition.

            Format your response as:
            Definition: [New Definition]
            END.
            """


def generate_definitions(text, pipe, concepts):
    all_definitions = {}
    with torch.no_grad():
        for concept in concepts:
            messages = [{"role": "user", "content": create_prompt(text, concept)}]

            response = pipe(messages, num_return_sequences=3, temperature=1.2, pad_token_id=pipe.llama_tokenizer.eos_token_id)

            definitions = []
            for idx in range(3):
                current_response = response[idx]["generated_text"][1]["content"]

                start_idx = current_response.find("Definition:") + len("Definition:")
                end_idx = current_response.find("END.")
                if start_idx < 0 or end_idx < 0:
                    return None

                definitions.append(current_response[start_idx:end_idx].strip())
            if len(definitions) > 0:
                all_definitions[concept] = definitions
    return all_definitions


def load_model():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipe = pipeline("text-generation", model=model_name, device_map="auto", torch_dtype=torch.bfloat16)
    return pipe


def generate_definition_dataset(start_idx, end_idx):
    abstracts = load_abstracts()
    concepts = load_concepts()
    keys = list(sorted(concepts.keys()))[start_idx:end_idx]

    pipe = load_model()

    bar = tqdm(total=len(keys), initial=0, desc="Generating definitions...", smoothing=0.5)
    for paper_idx in keys:
        bar.update(1)

        if os.path.exists(f"generated_definitions/{paper_idx}.json"):
            continue

        title = abstracts[paper_idx]['title']
        abstract = abstracts[paper_idx]['abstract']
        if title[-1].isalpha():
            title += "."
        text = f"{title} {abstract}"

        definitions = generate_definitions(text, pipe, concepts[paper_idx])

        with open(f"generated_definitions/{paper_idx}.json", "w") as f:
            json.dump(definitions, f)


if __name__ == '__main__':
    generate_definition_dataset(0, 2000)