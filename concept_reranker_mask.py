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
from itertools import cycle

device = "cuda" if torch.cuda.is_available() else "cpu"

class SpanPredModel(torch.nn.Module):

    def __init__(self, model):
        super(SpanPredModel, self).__init__()
        self.model = model
        self.dense_1 = torch.nn.Linear(768, 1)

    def forward(self, **kwargs):
        bert_out = self.model(**kwargs)["last_hidden_state"]

        span_pred_out = self.dense_1(bert_out[:, 1:-1])
        span_pred_out = span_pred_out.reshape(*span_pred_out.shape[:2])

        return span_pred_out

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


def collate_fn(dataloader, tokenizer, max_length=512):

    all_inputs = []
    all_masks = []

    while len(all_inputs) < 8:
        item = next(dataloader)
        abstract = item['abstract'][0]
        concept = item['concept'][0]
        definition = random.choice(item['definitions'][0])

        new_input = f"{definition} {tokenizer.sep_token} {abstract}"
        tokenizer_out = tokenizer(new_input, add_special_tokens=False)
        gt_array = np.zeros(len(tokenizer_out.tokens()))

        positions = [match.start() for match in re.finditer(re.escape(concept.lower()), new_input.lower())]
        positions = [x for x in positions if x > len(definition) + len(tokenizer.sep_token) + 1 and x < max_length-2]

        if len(positions) == 0:
            continue
        all_inputs.append(new_input)

        spans = [(s, s + len(concept)) for s in positions]

        for span in spans:
            token_indices = list(sorted(set([tokenizer_out.char_to_token(index) for index in range(*span)])))

            for idx in token_indices:
                gt_array[idx] = 1
        all_masks.append(gt_array)

    inputs = tokenizer(all_inputs, max_length=max_length, truncation=True, padding=True, return_tensors="pt").to(device)

    max_array_length = max(len(arr) for arr in all_masks)
    batch_masks = np.array([np.pad(arr, (0, max_array_length - len(arr)), mode='constant', constant_values=-1) for arr in all_masks])
    all_masks = torch.tensor(batch_masks[:, :max_length - 2], dtype=torch.float32, device=device)

    return inputs, all_masks


def evaluate_model(model, val_loader):
    model.eval()
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_samples = 0

    val_iter = iter(val_loader)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    first_batch = True
    with torch.no_grad():
        while True:
            try:
                inputs, masks = collate_fn(val_iter, tokenizer)
            except StopIteration:
                break

            predictions = torch.sigmoid(model(**inputs))
            # Convert predictions to binary using 0.5 threshold
            predictions = (predictions > 0.5).float()

            # Calculate metrics for each sample in batch
            for pred, mask in zip(predictions, masks):
                valid_mask = mask >= 0
                if valid_mask.sum() == 0:
                    continue

                pred = pred[valid_mask]
                mask = mask[valid_mask]

                # True positives, false positives, false negatives
                tp = ((pred == 1) & (mask == 1)).sum().float()
                fp = ((pred == 1) & (mask == 0)).sum().float()
                fn = ((pred == 0) & (mask == 1)).sum().float()

                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                total_precision += precision.item()
                total_recall += recall.item()
                total_f1 += f1.item() if f1 != 0 else 0
                total_samples += 1
            if first_batch:
                for idx in range(mask.shape[0]):
                    visualize_word_importance(list(zip(masks[idx].detach().cpu().numpy(), inputs.tokens(idx))))
                first_batch = False

    # Calculate averages
    avg_precision = total_precision / total_samples if total_samples > 0 else 0
    avg_recall = total_recall / total_samples if total_samples > 0 else 0
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0

    # Print metrics
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")

    return avg_f1


def train():
    model = AutoModel.from_pretrained("microsoft/deberta-base")
    model = SpanPredModel(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    abstracts = load_abstracts()
    concept_definitions = load_abstract_definitions()

    train_dataset, val_dataset = create_train_val_split(abstracts, concept_definitions)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    best_score = evaluate_model(model, val_loader)
    patience = 3
    patience_counter = 0

    train_iter = cycle(train_loader)
    for epoch in range(30):
        model.train()
        epoch_losses = []

        for _ in tqdm(range(1000), desc=f"Training (Epoch {epoch + 1})", file=sys.stdout):
            inputs, masks = collate_fn(train_iter, tokenizer)
            prediction = torch.sigmoid(model(**inputs))

            loss = -(torch.log(prediction[masks == 1]).sum(-1) + torch.log(1-prediction[masks==0]).sum(-1)) / (masks > 0).sum(-1)

            loss.mean().backward()
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