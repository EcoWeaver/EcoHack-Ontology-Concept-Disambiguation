import random
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from load_data import load_abstracts, load_concepts, load_ontology_data, load_abstract_definitions
from nltk.stem import PorterStemmer
import re
from visualize_text import visualize_word_importance
from itertools import cycle

device = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_concept_name(concept):
    ps = PorterStemmer()
    concept = concept.replace("-", " ")
    words = concept.split(" ")
    stemmed = [ps.stem(w) for w in words]
    return " ".join(stemmed)

class AbstractDataset(Dataset):
    def __init__(self, abstracts, definitions):
        self.data = []

        for id, concept_definitions in definitions.items():
            if concept_definitions is not None:
                for concept, defs in concept_definitions.items():
                        self.data.append({
                            'abstract': abstracts[id]["prediction_text"],
                            'concept': concept,
                            'definitions': defs,
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_train_val_split(abstracts, concept_definitions, val_size=40):
    valid_ids = list(concept_definitions.keys())

    val_ids = random.sample(valid_ids, min(val_size, len(valid_ids)))
    train_ids = [id for id in valid_ids if id not in val_ids]

    train_definitions = {id: concept_definitions[id] for id in train_ids}
    val_definitions = {id: concept_definitions[id] for id in val_ids}

    train_dataset = AbstractDataset(abstracts, train_definitions)
    val_dataset = AbstractDataset(abstracts, val_definitions)

    return train_dataset, val_dataset


def collate_fn(batch, tokenizer, max_length=512):
    all_inputs = []
    all_labels = []

    batch_definitions = {}
    for item in batch:
        batch_definitions[normalize_concept_name(item['concept'])] = random.choice(item["definitions"])

    for item in batch:
        abstract = item['abstract']
        concept = item['concept']
        definition = random.choice(item['definitions'])

        positions = [match.start() for match in re.finditer(re.escape(concept.lower()), abstract.lower())]

        if len(positions) == 0:
            continue

        position = random.choice(positions)
        new_abstract = f"{abstract[:position]}{tokenizer.sep_token} {concept} {tokenizer.sep_token}{abstract[position+len(concept):]}"

        positive_input_text = f"{definition} {tokenizer.sep_token} {new_abstract}"
        all_inputs.append(positive_input_text)
        all_labels.append(1)

        try:
            negative_definition = batch_definitions[random.choice([x for x in batch_definitions if x != normalize_concept_name(concept)])]
            negative_input_text = f"{negative_definition} {tokenizer.sep_token} {new_abstract}"

            all_inputs.append(negative_input_text)
            all_labels.append(0)
        except:
            print(batch_definitions.keys())
            print([x for x in batch_definitions if x != normalize_concept_name(concept)])
            quit()
    try:
        inputs = tokenizer(all_inputs, max_length=max_length, truncation=True, padding=True, return_tensors="pt").to(device)
    except:
        print("Inputs")
        print(all_inputs)
        quit()
    labels = torch.tensor(all_labels, dtype=torch.float, device=device)

    return inputs, labels


def evaluate_model(model, val_loader):
    model.eval()

    predictions = []
    labels = []
    losses = []

    with torch.no_grad():
        for inputs, batch_labels in tqdm(val_loader, desc="Evaluating"):
            outputs = model(**inputs).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, batch_labels)

            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            losses.append(loss.item())

    average_loss = np.mean(losses)
    accuracy = np.mean(((np.array(predictions) > 0.5) == np.array(labels)).astype(float))

    print(f"Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

    return average_loss


def train():
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=1).to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    abstracts = load_abstracts()
    concept_definitions = load_abstract_definitions()

    train_dataset, val_dataset = create_train_val_split(abstracts, concept_definitions)
    random.shuffle(val_dataset.data)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    best_score = evaluate_model(model, val_loader)
    patience = 3
    patience_counter = 0

    train_iter = cycle(train_loader)
    for epoch in range(30):
        model.train()
        epoch_losses = []

        for _ in tqdm(range(1000), desc=f"Training (Epoch {epoch + 1})", file=sys.stdout):
            inputs, labels = next(train_iter)
            outputs = model(**inputs).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())

        average_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} average loss: {average_loss}")

        current_score = evaluate_model(model, val_loader)

        if current_score < best_score:
            print("New best score! Saving model...")
            torch.save(model.state_dict(), "models/concept_reranker.pkl")
            best_score = current_score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"No improvement for {patience} epochs. Stopping training.")
                break


if __name__ == "__main__":
    train()