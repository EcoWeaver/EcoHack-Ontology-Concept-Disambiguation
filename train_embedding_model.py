import torch
from transformers import AutoModel, AutoTokenizer

def train():
    model = AutoModel.from_pretrained("microsoft/deberta-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    model.load_state_dict("models/pretrained_model.pkl")

    model = model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    for epoch in range(20):
        for batch_idx in range(1000):


            distances = torch.cdist(embeddings_1, embeddings_2, p=2)

            positive_distances = distances.diag().unsqueeze(-1).repeat(1, SENTENCE_BATCH_SIZE)
            combined_distances = torch.cat([positive_distances.unsqueeze(-1), distances.unsqueeze(-1)], dim=-1)

            scores = torch.relu(combined_distances[:, :, 0] - combined_distances[:, :, 1] + 1)
            off_diagonal_mask = ~torch.eye(scores.size(0), dtype=torch.bool, device="cuda")
            off_diagonal_entries = scores[off_diagonal_mask]

            loss = (1 / grad_accumulation_steps) * off_diagonal_entries.mean()
            loss += 1 / 250 * torch.norm(embeddings_1, dim=-1).mean() * (current_batch_scores / target_n_scores)

            loss.backward()