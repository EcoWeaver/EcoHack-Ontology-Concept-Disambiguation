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
            result[index] = {"title": title, "abstract": abstract, "index": index}
    return result

def load_concepts():
    result = {}
    for filename in tqdm(os.listdir("detected_concepts")):
        with open(f"detected_concepts/{filename}", "r") as f:
            concepts = json.load(f)
            
            index = int(filename[:-5])
            result[index] = concepts
    return result