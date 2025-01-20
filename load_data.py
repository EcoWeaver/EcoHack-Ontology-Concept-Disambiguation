import os

def load_abstracts():
    result = {}
    for filename in os.listdir("unlabeled abstracts"):
        with open(f"unlabeled abstracts/{filename}", "r") as f:
            text = f.read().strip()
            title, abstract, doi = text.split("\n")

            index = int(filename[:-4])
            result[index] = {"title": title, "abstract": abstract, "index": index}
    return result