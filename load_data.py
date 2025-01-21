import os
import json
from tqdm import tqdm

def load_abstracts():
    result = {}
    for filename in tqdm(os.listdir("unlabeled abstracts")):
        with open(f"unlabeled abstracts/{filename}", "r") as f:
            text = f.read().strip()
            title, abstract, doi = text.split("\n")

            index = int(filename[:-4])

            if title[-1].isalpha():
                new_title = title + "."
            else:
                new_title = title
            text = f"{new_title} {abstract}"
            result[index] = {"title": title, "abstract": abstract, "index": index, "prediction_text": text}
    return result

def load_concepts():
    result = {}
    for filename in tqdm(os.listdir("detected_concepts")):
        with open(f"detected_concepts/{filename}", "r") as f:
            concepts = json.load(f)
            
            index = int(filename[:-5])
            result[index] = concepts
    return result