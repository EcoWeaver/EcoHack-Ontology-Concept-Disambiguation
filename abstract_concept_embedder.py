import random
import sys
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from load_data import load_abstracts, load_concepts, load_ontology_data, load_abstract_definitions
from nltk.stem import PorterStemmer

device = "cuda" if torch.cuda.is_available() else "cpu"

class TokenEmbeddingModel(torch.nn.Module):
    def __init__(self, model):
        super(TokenEmbeddingModel, self).__init__()
        self.model = model
        self.projection = torch.nn.Linear(768, 768)  # Assuming DeBERTa base with 768 dimensions

    def forward(self, **kwargs):
        bert_out = self.model(**kwargs)["last_hidden_state"]
        token_embeddings = self.projection(bert_out[:, 1:-1])  # Remove CLS and SEP tokens
        return token_embeddings


class ConceptEmbeddingDataset(Dataset):
    def __init__(self, abstracts, concepts, concept_embeddings):
        self.abstracts = {}
        self.concepts = {}
        self.concept_embeddings = {}

        # Only keep samples that have both concepts and embeddings
        for abstract_id in abstracts:
            if abstract_id in concepts:
                valid_concepts = []
                valid_embeddings = []

                for concept in concepts[abstract_id]:
                    if concept in concept_embeddings:
                        valid_concepts.append(concept)
                        valid_embeddings.append(concept_embeddings[concept])

                if valid_concepts:  # Only keep if at least one concept has an embedding
                    self.abstracts[abstract_id] = abstracts[abstract_id]["prediction_text"]
                    self.concepts[abstract_id] = valid_concepts
                    self.concept_embeddings[abstract_id] = valid_embeddings

        self.indices = list(self.abstracts.keys())

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        abstract_id = self.indices[idx]
        return {
            'abstract': self.abstracts[abstract_id],
            'concepts': self.concepts[abstract_id],
            'embeddings': self.concept_embeddings[abstract_id]
        }


def get_concept_embeddings():
    # Load pre-trained definition embedding model
    definition_model = AutoModel.from_pretrained("microsoft/deberta-base").to(device)
    definition_model.load_state_dict(torch.load("models/definition_embedding_model.pkl"))
    definition_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    # Load all definitions
    abstract_definitions = load_abstract_definitions()

    concept_embeddings = {}

    with torch.no_grad():
        for id in tqdm(abstract_definitions, desc="Computing concept embeddings"):
            abstract_embeddings = {}
            for concept in abstract_definitions[id]:
                definitions = abstract_definitions[id][concept]

                if len(definitions) == 0:
                    continue

                inputs = tokenizer(definitions, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                embedding = definition_model(**inputs)["last_hidden_state"][:, 0, :].mean(0)
                abstract_embeddings[concept].append(embedding.detach())

            concept_embeddings[id] = abstract_embeddings

    return concept_embeddings


def collate_fn(batch, tokenizer, max_length=512):
    all_abstracts = []
    all_concepts = []
    all_embeddings = []

    for item in batch:
        abstract = item["abstract"]
        all_abstracts.append(abstract)
        all_concepts.append(item["concepts"])
        all_embeddings.append(item["embeddings"])

    inputs = tokenizer(all_abstracts, max_length=max_length, truncation=True,
                       padding=True, return_tensors="pt").to(device)

    # Create embedding target tensor
    max_length = inputs['input_ids'].shape[1] - 2  # Account for CLS and SEP tokens
    batch_size = len(batch)

    target_embeddings = torch.zeros((batch_size, max_length, 768), device=device)
    target_masks = torch.zeros((batch_size, max_length), device=device)

    # Map concepts to their token positions
    for batch_idx, (abstract, concepts, embeddings) in enumerate(zip(all_abstracts, all_concepts, all_embeddings)):
        tokenizer_out = tokenizer([abstract], add_special_tokens=False)

        for concept, embedding in zip(concepts, embeddings):
            matches = [match.start() for match in re.finditer(re.escape(concept.lower()), abstract.lower())]
            spans = [(s, s + len(concept)) for s in matches]

            for span in spans:
                token_indices = list(sorted(set([tokenizer_out.char_to_token(index)
                                                 for index in range(*span) if
                                                 tokenizer_out.char_to_token(index) is not None])))
                if token_indices and token_indices[0] < max_length:
                    target_embeddings[batch_idx, token_indices[0]] = embedding
                    target_masks[batch_idx, token_indices[0]] = 1

                    for other_index in token_indices[1:]:
                        if other_index < max_length:
                            target_masks[batch_idx, other_index] = 1

    return inputs, target_embeddings, target_masks


def train():
    model = AutoModel.from_pretrained("microsoft/deberta-base")
    model = TokenEmbeddingModel(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    # Load data
    abstracts = load_abstracts()
    concepts = load_concepts()
    concept_embeddings = get_concept_embeddings()

    # Create dataset
    dataset = ConceptEmbeddingDataset(abstracts, concepts, concept_embeddings)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, tokenizer))

    best_loss = float('inf')
    for epoch in range(30):
        model.train()
        epoch_losses = []

        for inputs, target_embeddings, target_masks in tqdm(dataloader,
                                                            desc=f"Training (Epoch {epoch + 1})",
                                                            file=sys.stdout):
            predicted_embeddings = model(**inputs)

            # Compute loss only for tokens that have target embeddings
            loss = torch.nn.functional.mse_loss(
                predicted_embeddings[target_masks == 1],
                target_embeddings[target_masks == 1]
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())

        average_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} average loss: {average_loss}")

        if average_loss < best_loss:
            print("New best!")
            torch.save(model.state_dict(), "models/token_embedding_model.pkl")
            best_loss = average_loss


if __name__ == "__main__":
    train()