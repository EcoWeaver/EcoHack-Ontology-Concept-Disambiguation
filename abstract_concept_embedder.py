import random
import sys
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from load_data import load_abstracts, load_concepts, load_ontology_data, load_abstract_definitions
from nltk.stem import PorterStemmer
import re
from visualize_text import visualize_word_importance

device = "cuda" if torch.cuda.is_available() else "cpu"

class TokenEmbeddingModel(torch.nn.Module):
    def __init__(self, model):
        super(TokenEmbeddingModel, self).__init__()
        self.model = model

    def forward(self, **kwargs):
        bert_out = self.model(**kwargs)["last_hidden_state"][:, 1:-1]
        return bert_out


class ConceptEmbeddingDataset(Dataset):
    def __init__(self, abstracts, concept_embeddings):
        self.abstracts = {}
        self.concept_embeddings = {}

        # Only keep samples that have both concepts and embeddings
        for abstract_id in concept_embeddings:
            self.abstracts[abstract_id] = abstracts[abstract_id]["prediction_text"]
            self.concept_embeddings[abstract_id] = concept_embeddings[abstract_id]

        self.indices = list(self.abstracts.keys())

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        abstract_id = self.indices[idx]
        return {
            'abstract': self.abstracts[abstract_id],
            'embeddings': self.concept_embeddings[abstract_id]
        }


def get_concept_embeddings():
    cache_path = "models/concept_embeddings.pkl"
    try:
        concept_embeddings = torch.load(cache_path)
        print(f"Loaded cached concept embeddings from {cache_path}")
        return concept_embeddings
    except FileNotFoundError:
        print("No cached embeddings found. Computing embeddings...")

    definition_model = AutoModel.from_pretrained("microsoft/deberta-base").to(device)
    definition_model.load_state_dict(torch.load("models/definition_embedding_model.pkl"))
    definition_model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    abstract_definitions = load_abstract_definitions()
    #ids = list(abstract_definitions.keys())[:100]
    #abstract_definitions = {idx: abstract_definitions[idx] for idx in ids}
    concept_embeddings = {}

    with torch.no_grad():
        for id in tqdm(abstract_definitions, desc="Computing concept embeddings"):
            abstract_embeddings = {}
            if abstract_definitions[id] is not None:
                for concept in abstract_definitions[id]:
                    definitions = abstract_definitions[id][concept]

                    if len(definitions) == 0:
                        continue

                    inputs = tokenizer(definitions, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                    embedding = definition_model(**inputs)["last_hidden_state"][:, 0, :].mean(0)
                    abstract_embeddings[concept] = embedding.detach()

                concept_embeddings[id] = abstract_embeddings

    torch.save(concept_embeddings, cache_path)
    print(f"Saved concept embeddings to {cache_path}")

    return concept_embeddings


def collate_fn(batch, tokenizer, max_length=512):
    all_abstracts = []
    all_embeddings = []

    for item in batch:
        abstract = item["abstract"]
        all_abstracts.append(abstract)
        all_embeddings.append(item["embeddings"])

    inputs = tokenizer(all_abstracts, max_length=max_length, truncation=True, padding=True, return_tensors="pt").to(device)

    max_length = inputs['input_ids'].shape[1] - 2  # Account for CLS and SEP tokens
    batch_size = len(batch)

    target_embeddings = torch.zeros((batch_size, max_length, 768), device=device)
    target_masks = torch.zeros((batch_size, max_length), device=device)
    ids_to_concepts = []

    for batch_idx, (abstract, embeddings) in enumerate(zip(all_abstracts, all_embeddings)):
        tokenizer_out = tokenizer([abstract], add_special_tokens=False)
        mask_id_to_concepts = {}
        for concept, embedding in embeddings.items():

            matches = [match.start() for match in re.finditer(re.escape(concept.lower()), abstract.lower())]
            spans = [(s, s + len(concept)) for s in matches]

            for span in spans:
                token_indices = list(sorted(set([tokenizer_out.char_to_token(index) for index in range(*span)])))
                if token_indices and token_indices[0] < max_length:
                    target_embeddings[batch_idx, token_indices[0]] = embedding
                    target_masks[batch_idx, token_indices[0]] = 1
                    mask_id_to_concepts[token_indices[0]] = concept
        ids_to_concepts.append(mask_id_to_concepts)

    #for mask, abstract in zip(target_masks.detach().cpu().numpy(), all_abstracts):
    #    visualize_word_importance(list(zip(mask, tokenizer.tokenize(abstract))))
    return inputs, target_embeddings, target_masks, ids_to_concepts


def normalize_concept_name(concept):
    ps = PorterStemmer()
    concept = concept.replace("-", " ")
    words = concept.split(" ")
    stemmed = [ps.stem(w) for w in words]
    return " ".join(stemmed)


def get_distinct_concepts(concept_embeddings, num_concepts=500):
    concept_to_embedding = {}
    for abstract_embeddings in concept_embeddings.values():
        for concept, embedding in abstract_embeddings.items():
            normalized = normalize_concept_name(concept)
            if normalized not in concept_to_embedding:
                concept_to_embedding[normalized] = embedding

    concepts = list(concept_to_embedding.keys())
    if len(concepts) > num_concepts:
        concepts = random.sample(concepts, num_concepts)

    return {concept: concept_to_embedding[concept] for concept in concepts}


def create_train_val_split(abstracts, concept_embeddings, val_size=40):
    valid_ids = list(concept_embeddings.keys())

    val_ids = random.sample(valid_ids, min(val_size, len(valid_ids)))
    train_ids = [id for id in valid_ids if id not in val_ids]

    train_embeddings = {id: concept_embeddings[id] for id in train_ids}
    val_embeddings = {id: concept_embeddings[id] for id in val_ids}

    train_dataset = ConceptEmbeddingDataset(abstracts, train_embeddings)
    val_dataset = ConceptEmbeddingDataset(abstracts, val_embeddings)

    return train_dataset, val_dataset


def evaluate_model(model, val_loader, distinct_concepts, device):
    model.eval()
    total_precision = 0
    total_rank = 0
    total_samples = 0

    # Convert distinct concepts to tensor for efficient comparison
    concept_embeddings = torch.stack(list(distinct_concepts.values())).to(device)
    negative_concept_names = list(distinct_concepts.keys())
    with torch.no_grad():
        for inputs, target_embeddings, target_masks, mask_ids_to_concepts in tqdm(val_loader, desc="Evaluating"):
            predicted_embeddings = model(**inputs)

            batch_size = predicted_embeddings.shape[0]
            for b in range(batch_size):
                mask_positions = torch.where(target_masks[b] == 1)[0]
                if len(mask_positions) == 0:
                    continue

                sample_precision = 0
                sample_rank_sum = 0
                num_concepts = len(mask_positions)

                for pos in mask_positions:
                    concept_name = normalize_concept_name(mask_ids_to_concepts[b][pos.item()])
                    try:
                        concept_name_index = negative_concept_names.index(concept_name)
                        current_concept_embeddings = torch.cat([concept_embeddings[:concept_name_index], concept_embeddings[concept_name_index + 1:]])
                    except:
                        current_concept_embeddings = concept_embeddings

                    pred_embedding = predicted_embeddings[b, pos]
                    true_embedding = target_embeddings[b, pos]

                    distances = torch.norm(current_concept_embeddings - pred_embedding.unsqueeze(0), dim=1)
                    true_distance = torch.norm(true_embedding - pred_embedding)

                    all_distances = torch.cat([distances, true_distance.unsqueeze(0)])
                    sorted_indices = torch.argsort(all_distances)
                    rank = (sorted_indices == len(distances)).nonzero().item() + 1

                    is_closest = (rank == 1)

                    sample_precision += float(is_closest)
                    sample_rank_sum += rank

                sample_precision /= num_concepts
                sample_avg_rank = sample_rank_sum / num_concepts

                total_precision += sample_precision
                total_rank += sample_avg_rank
                total_samples += 1

    avg_precision = total_precision / total_samples if total_samples > 0 else 0
    avg_rank = total_rank / total_samples if total_samples > 0 else 0
    print(f"Precision: {avg_precision:.4f}\nAverage Rank: {avg_rank:.2f}")

    return avg_precision, avg_rank


def train():
    model = AutoModel.from_pretrained("microsoft/deberta-base")
    model = TokenEmbeddingModel(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    abstracts = load_abstracts()
    concept_embeddings = get_concept_embeddings()

    distinct_concepts = get_distinct_concepts(concept_embeddings)

    train_dataset, val_dataset = create_train_val_split(abstracts, concept_embeddings)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    _, best_score = evaluate_model(model, val_loader, distinct_concepts, device)
    patience = 3
    patience_counter = 0

    for epoch in range(30):
        model.train()
        epoch_losses = []

        for inputs, target_embeddings, target_masks, _ in tqdm(train_loader, desc=f"Training (Epoch {epoch + 1})", file=sys.stdout):
            predicted_embeddings = model(**inputs)
            loss = torch.nn.functional.mse_loss(predicted_embeddings[target_masks == 1], target_embeddings[target_masks == 1])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())

        average_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} average loss: {average_loss}")

        _, current_score = evaluate_model(model, val_loader, distinct_concepts, device)

        if current_score < best_score:
            print("New best rank! Saving model...")
            torch.save(model.state_dict(), "models/abstract_token_embedding_model.pkl")
            best_score = current_score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"No improvement for {patience} epochs. Stopping training.")
                break


if __name__ == "__main__":
    train()