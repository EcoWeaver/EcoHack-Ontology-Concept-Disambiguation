import random
import sys

from scipy.integrate import tplquad
from sklearn.metrics import ndcg_score
import torch
from transformers import AutoModel, AutoTokenizer
from load_data import load_abstracts, load_concepts
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from tqdm import tqdm
from visualize_text import visualize_word_importance

device = "cuda" if torch.cuda.is_available() else "cpu"

class SpanPredModel(torch.nn.Module):

    def __init__(self, model):
        super(SpanPredModel, self).__init__()
        self.model = model
        self.dense_1 = torch.nn.Linear(768, 3)

    def forward(self, **kwargs):
        bert_out = self.model(**kwargs)["last_hidden_state"]

        span_pred_out = self.dense_1(bert_out[:, 1:-1])
        span_pred_out = span_pred_out.reshape(*span_pred_out.shape[:2], 3)

        return span_pred_out

class AbstractDataset(Dataset):
    def __init__(self, abstracts, concepts):
        self.abstracts = {}
        self.concepts = {}

        for abstract_id in abstracts:
            if abstract_id in concepts:
                self.abstracts[abstract_id] = abstracts[abstract_id]["prediction_text"]
                self.concepts[abstract_id] = concepts[abstract_id]
        self.indices = list(self.abstracts.keys())

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        return {
            'abstract': self.abstracts[self.indices[idx]],
            'concepts': self.concepts[self.indices[idx]]
        }


def collate_fn(batch, tokenizer, max_length=512):
    all_abstracts = []
    all_labels = []

    for item in batch:
        abstract = item["abstract"]
        all_abstracts.append(abstract)
        concepts = item["concepts"]

        tokenizer_out = tokenizer([abstract], add_special_tokens=False)
        gt_array = np.zeros(len(tokenizer_out.tokens()))

        for concept in concepts:
            if len(concept) < 3:
                continue

            matches = [match.start() for match in re.finditer(re.escape(concept.lower()), abstract.lower())]
            spans = [(s, s + len(concept)) for s in matches]

            for span in spans:
                token_indices = list(sorted(set([tokenizer_out.char_to_token(index) for index in range(*span)])))
                gt_array[token_indices[0]] = 1
                for other_index in token_indices[1:]:
                    gt_array[other_index] = 2
        all_labels.append(gt_array)

    inputs = tokenizer(all_abstracts, max_length=max_length, truncation=True, padding=True, return_tensors="pt").to(device)

    max_array_length = max(len(arr) for arr in all_labels)
    batch_masks = np.array([np.pad(arr, (0, max_array_length - len(arr)), mode='constant', constant_values=-1) for arr in all_labels])
    masks = torch.tensor(batch_masks[:, :max_length - 2], dtype=torch.float32, device=device)

    return inputs, masks

def evaluate(model, dataloader):
    model.eval()
    scores = []
    first_batch = True
    for inputs, masks in tqdm(dataloader, desc="Evaluating", file=sys.stdout):
        prediction = model(**inputs)
        prediction = torch.softmax(prediction, dim=-1)
        positive_scores = torch.sum(prediction[:, :, 1:], dim=-1).detach().cpu().numpy()

        for i in range(masks.shape[0]):
            scores.append(ndcg_score((masks[i] > 0).cpu().numpy().astype("float").reshape(1, -1), positive_scores[i].reshape(1, -1)))

        if first_batch:
            predicted_values = torch.sum(prediction[0, :, 1:], dim=-1).detach().cpu().numpy().tolist()
            tokens = inputs.tokens(0)[1:-1]
            gt = masks[0].detach().cpu().numpy()
            score_list = [(x, y) for x, y, z in zip(predicted_values, tokens, gt) if z >= 0]
            visualize_word_importance(score_list)
            first_batch = False
    average_score = np.mean(scores)
    print(f"Average ndcg: {average_score}")
    return average_score

def train():
    model = AutoModel.from_pretrained("microsoft/deberta-base")
    model = SpanPredModel(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    abstracts = load_abstracts()
    concepts = load_concepts()

    all_ids = list(concepts.keys())
    random.shuffle(all_ids)
    val_ids = all_ids[:100]
    train_ids = all_ids[100:]

    train_dataset = AbstractDataset(abstracts, {x: concepts[x] for x in train_ids})
    val_dataset = AbstractDataset(abstracts, {x: concepts[x] for x in val_ids})
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    val_score = evaluate(model, val_loader)

    for epoch in range(30):
        model.train()
        for inputs, masks in tqdm(train_loader, desc=f"Training (Epoch {epoch+1})", file=sys.stdout):

            prediction = model(**inputs)
            prediction = torch.softmax(prediction, dim=-1)

            loss = torch.zeros(masks.shape[0], device=device)
            for i in range(1, 3):
                label_mask = (masks == i).to(torch.float)
                loss += (-torch.log(prediction[:, :, i]) * label_mask).sum(-1) / (masks >= 0).sum(-1)
            loss = loss.mean()

            label_mask = (masks >= 0).to(torch.float)
            loss += torch.square((prediction[:, :, 1:].sum(-1) * label_mask).sum(-1) / label_mask.sum(-1)).mean()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.empty_cache()
        new_val_score = evaluate(model, val_loader)
        if new_val_score > val_score:
            print("New best!")
            torch.save(model.state_dict(), "models/concept_identifier.pkl")
            val_score = new_val_score


train()