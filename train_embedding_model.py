import random
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from load_data import load_ontology_data, load_abstract_definitions
import re
from nltk.stem import PorterStemmer

training_definitions = {}
validation_definitions = {}

def load_data():
    global training_definitions, validation_definitions

    ps = PorterStemmer()

    def normalize_concept_name(concept):
        concept = concept.replace("-", " ")
        words = concept.split(" ")
        stemmed = [ps.stem(w) for w in words]
        return " ".join(stemmed)

    ontology_data = load_ontology_data()
    abstract_definitions = load_abstract_definitions()

    all_definitions = {}
    for concept in ontology_data:
       key = normalize_concept_name(concept)
       if key in abstract_definitions:
           all_definitions[key].append(ontology_data[concept]["generated_definitions"] + ontology_data[concept]["definitions"])
       else:
           all_definitions[key] = [ontology_data[concept]["generated_definitions"] + ontology_data[concept]["definitions"]]

    for id in abstract_definitions:
        if abstract_definitions[id] is not None:
            for concept in abstract_definitions[id]:
                key = normalize_concept_name(concept)
                if key in all_definitions:
                    all_definitions[key].append(abstract_definitions[id][concept])
                else:
                    all_definitions[key] = [abstract_definitions[id][concept]]

    for _ in range(50):
        while True:
            key = random.choice(list(all_definitions.keys()))
            if key not in validation_definitions:
                idx = random.randrange(0, len(all_definitions[key]))
                definitions = all_definitions[key].pop(idx)
                validation_definitions[key] = definitions[:2]
                if len(all_definitions[key]) == 0:
                    del all_definitions[key]
                break
    training_definitions = all_definitions

def batch_generator(batch_size=32):

    keys = []
    while True:
        batch = []
        for _ in range(batch_size):
            while True:
                if len(keys) == 0:
                    keys = list(training_definitions.keys())
                    random.shuffle(keys)
                key = keys.pop()

                definition_selection = random.choice(training_definitions[key])
                if len(definition_selection) >= 2:
                    definitions = random.sample(definition_selection, k=2)
                    batch.append(definitions)
                    break
        yield batch


def evaluate(model, tokenizer):
    model.eval()
    all_embeddings_1 = []
    all_embeddings_2 = []

    with torch.no_grad():
        for concept in validation_definitions:
            d_1, d_2 = validation_definitions[concept]

            tokens_1 = tokenizer(d_1, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            tokens_2 = tokenizer(d_2, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")

            emb_1 = model(**tokens_1)["last_hidden_state"][:, 0, :]
            emb_2 = model(**tokens_2)["last_hidden_state"][:, 0, :]

            all_embeddings_1.append(emb_1)
            all_embeddings_2.append(emb_2)

    embeddings_1 = torch.cat(all_embeddings_1, dim=0)
    embeddings_2 = torch.cat(all_embeddings_2, dim=0)

    distances = torch.cdist(embeddings_1, embeddings_2, p=2)

    positive_distances = distances.diag().unsqueeze(-1).repeat(1, embeddings_1.shape[0])
    combined_distances = torch.cat([positive_distances.unsqueeze(-1), distances.unsqueeze(-1)], dim=-1)

    scores = torch.relu(combined_distances[:, :, 0] - combined_distances[:, :, 1] + 1)
    off_diagonal_mask = ~torch.eye(scores.size(0), dtype=torch.bool, device="cuda")
    off_diagonal_entries = scores[off_diagonal_mask]

    loss = off_diagonal_entries.mean()

    print(f"Loss: {loss.item()}")
    return loss.item()

def train():
    model = AutoModel.from_pretrained("microsoft/deberta-base").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    model.load_state_dict(torch.load("models/pretrained_model.pkl"))

    model = model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    load_data()
    generator = batch_generator()

    best_score = evaluate(model, tokenizer)
    for epoch in range(20):
        model.train()

        for batch_idx in tqdm(range(1000)):
            batch = next(generator)

            d_1 = [x[0] for x in batch]
            d_2 = [x[1] for x in batch]

            batch_1 = tokenizer(d_1, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            batch_2 = tokenizer(d_2, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")

            embeddings_1 = model(**batch_1)["last_hidden_state"][:,0,:]
            embeddings_2 = model(**batch_2)["last_hidden_state"][:,0,:]

            distances = torch.cdist(embeddings_1, embeddings_2, p=2)

            positive_distances = distances.diag().unsqueeze(-1).repeat(1, embeddings_1.shape[0])
            combined_distances = torch.cat([positive_distances.unsqueeze(-1), distances.unsqueeze(-1)], dim=-1)

            scores = torch.relu(combined_distances[:, :, 0] - combined_distances[:, :, 1] + 1)
            off_diagonal_mask = ~torch.eye(scores.size(0), dtype=torch.bool, device="cuda")
            off_diagonal_entries = scores[off_diagonal_mask]

            loss = off_diagonal_entries.mean()
            loss += 1 / 250 * torch.norm(embeddings_1, dim=-1).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        score = evaluate(model, tokenizer)
        if score <= best_score:
            best_score = score
            print("New best!")
            torch.save(model.state_dict(), "models/definition_embedding_model.pkl")
train()