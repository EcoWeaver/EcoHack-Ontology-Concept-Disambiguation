import random
import sys
import torch
import os
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from blingfire import text_to_sentences
from nltk.stem import PorterStemmer
from itertools import cycle
from collections import defaultdict
from load_data import load_abstracts, load_abstract_definitions
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
target_ids = {label: torch.tensor(llama_tokenizer.encode(output)[1], device=device, dtype=torch.long).reshape(1, ) for label, output in [(0, " no"), (1, " yes")]}

# Normalizing concept names
def normalize_concept_name(concept):
    ps = PorterStemmer()
    concept = concept.replace("-", " ")
    words = concept.split(" ")
    stemmed = [ps.stem(w) for w in words]
    return " ".join(stemmed)

# Dataset class
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

# Create train-validation split
def create_train_val_split(abstracts, concept_definitions, val_size=40):
    valid_ids = list(concept_definitions.keys())
    val_ids = random.sample(valid_ids, min(val_size, len(valid_ids)))
    train_ids = [id for id in valid_ids if id not in val_ids]

    train_definitions = {id: concept_definitions[id] for id in train_ids}
    val_definitions = {id: concept_definitions[id] for id in val_ids}

    train_dataset = AbstractDataset(abstracts, train_definitions)
    val_dataset = AbstractDataset(abstracts, val_definitions)

    return train_dataset, val_dataset

# Prepares input for the model based on the chat format
def format_input(definition, concept, sentence):
    chat = [
        {"role": "system", "content": "You are a specialist in ontologies and concepts in scientific texts. Your task is, to assess, if a given definition from an ontology concept matches a specific concept that occurred in a sentence in a scientific abstract. You are given that sentence to know the context that the word appeared in."},
        {"role": "user", "content": f"Task: Determine if the following definition matches the given concept, which occurred in the sentence that is provided.\n\nDefinition: {definition}\nConcept: {concept}\nSentence: {sentence}\nPlease respond in the following format:\nAnswer: [yes/no]"},
        {"role": "assistant", "content": "Answer:"}
    ]
    return llama_tokenizer.apply_chat_template(chat, tokenize=False)

# Collate function for DataLoader
def collate_fn(batch, definition_dict):
    inputs, labels = [], []
    for item in batch:
        abstract = item['abstract']
        concept = item['concept']
        normalized_concept = normalize_concept_name(concept)

        sentences = text_to_sentences(abstract).split("\n")
        concept_sentences = [s for s in sentences if concept.lower() in s.lower()]
        if len(concept_sentences) == 0:
            return None, None
        sentence = random.choice(concept_sentences)

        # Decide randomly whether to use a positive or negative example
        if random.random() > 0.5:  # Positive example
            definition = random.choice(item['definitions'])
            label = 1
        else:  # Negative example
            negative_definitions = [d for key, defs in definition_dict.items() if key != normalized_concept for d in defs]
            definition = random.choice(negative_definitions)
            label = 0

        inputs.append(format_input(definition, concept, sentence))
        labels.append(label)

    inputs = llama_tokenizer(inputs, return_tensors="pt").to(device)
    return inputs, torch.tensor(labels, dtype=torch.float, device=device)

# Train function
def train():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    abstracts = load_abstracts()
    concept_definitions = load_abstract_definitions()

    train_dataset, val_dataset = create_train_val_split(abstracts, concept_definitions)

    definition_dict = defaultdict(list)
    for item in train_dataset:
        for definition in item['definitions']:
            definition_dict[normalize_concept_name(item['concept'])].append(definition)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda b: collate_fn(b, definition_dict))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda b: collate_fn(b, definition_dict))

    best_score = evaluate_model(model, val_loader)
    patience = 3
    patience_counter = 0
    gradient_accumulation_steps = 8

    train_iter = cycle(train_loader)
    for epoch in range(30):
        model.train()
        epoch_losses = []

        optimizer.zero_grad()
        for step in tqdm(range(200), desc=f"Training Epoch {epoch + 1}"):
            acc_idx = 0
            while acc_idx < gradient_accumulation_steps:
                inputs, labels = next(train_iter)
                if inputs is None: continue
                logits = model(**inputs).logits[0][-2]
                loss = torch.nn.functional.cross_entropy(logits.reshape(1, -1), target_ids[labels[0].item()])
                loss = loss / gradient_accumulation_steps
                loss.backward()
                acc_idx += 1

            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())

        average_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} Average Loss: {average_loss}")

        current_score = evaluate_model(model, val_loader)

        if current_score < best_score:
            print("New best score! Saving model...")
            torch.save(model.state_dict(), "models/concept_reranker_LLM.pkl")
            best_score = current_score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"No improvement for {patience} epochs. Stopping training.")
                break

# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    all_losses = []
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            if inputs is None:
                continue
            logits = model(**inputs).logits[0][-2]

            if logits[target_ids[0]] > logits[target_ids[1]]:
                all_predictions.append(0)
            else:
                all_predictions.append(1)
            all_labels.append(labels.item())
            loss = torch.nn.functional.cross_entropy(logits.reshape(1, -1), target_ids[labels[0].item()])
            all_losses.append(loss.item())

    average_loss = np.mean(all_losses)
    print(f"Loss: {average_loss:.3f}")
    accuracy = float(np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels))
    print(f"Accuracy: {accuracy:.3f}")
    return average_loss

if __name__ == "__main__":
    train()
