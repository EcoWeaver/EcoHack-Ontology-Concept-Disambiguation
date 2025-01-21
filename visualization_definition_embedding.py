

import torch
from transformers import AutoModel, AutoTokenizer

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained model and tokenizer
model_name = "microsoft/deberta-base"  # Replace with the model you used
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Load the weights from the .pkl file
weights_path = "/content/drive/MyDrive/definition_embedding_model.pkl"  # Replace with the actual path
state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Function to run prediction
def predict(abstract, concept, definition):
    position = abstract.lower().find(concept.lower())
    if position != -1:
        new_abstract = f"{abstract[:position]} {tokenizer.sep_token} {concept} {tokenizer.sep_token}{abstract[position+len(concept):]}"
    else:
        new_abstract = abstract  # Fallback if concept is not found

    input_text = f"{definition} {tokenizer.sep_token} {new_abstract}"

    # Tokenize input
    inputs = tokenizer(input_text, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Example: Pooler output or last hidden state
    pooled_output = outputs.last_hidden_state.mean(dim=1)  # Taking the mean across sequence dimension

    return pooled_output

import pandas as pd

# Load the CSV file
file_path = '/content/INBIOV2directparent.csv'
df = pd.read_csv(file_path)

# Drop rows where both 'classDefinition' and 'classComment' are NaN
filtered_df = df.dropna(subset=['classDefinition', 'classComment'], how='all')

# Create the dictionary using classLabel as keys
# Use classDefinition if available, otherwise fall back to classComment
class_dict = {}

for _, row in filtered_df.iterrows():
    class_label = row['classLabel']
    class_definition = row['classDefinition']
    class_comment = row['classComment']
    class_dict[class_label] = class_definition if pd.notna(class_definition) else class_comment

# Display the dictionary (for debugging or further processing)
print(class_dict)

#output_file = 'class_label_dictionary_filtered.json'
#with open(output_file, 'w') as f:
#    import json
#    json.dump(class_dict, f, indent=4)
#print(f"Filtered dictionary has been saved to {output_file}")

import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained model and tokenizer
model_name = "microsoft/deberta-base"  # Replace with the model you used
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Set the model to evaluation mode
model.eval()

# List of concepts and definitions


import numpy as np

# Function to compute embeddings for definitions
def compute_embeddings(definitions):
    embeddings = []
    for concept, definition in definitions.items():
        input_text = definition  # Use definition as input
        inputs = tokenizer(input_text, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()  # Mean pooling
            embeddings.append(embedding)

    # Convert the list of embeddings to a NumPy array
    return np.array(embeddings)

# Generate embeddings for all definitions
embeddings = compute_embeddings(class_dict)

# Apply t-SNE with perplexity less than n_samples
tsne = TSNE(n_components=2, perplexity=5, random_state=42) # Changed perplexity to 5, which is less than 10 (number of samples)
reduced_embeddings = tsne.fit_transform(embeddings)

import plotly.express as px
import pandas as pd

# Prepare data for Plotly visualization
plot_data = {
    "Concept": list(class_dict.keys()),
    "x": reduced_embeddings[:, 0],
    "y": reduced_embeddings[:, 1],
}

# Create a DataFrame for Plotly
plot_df = pd.DataFrame(plot_data)

# Create interactive Plotly visualization
fig = px.scatter(
    plot_df,
    x="x",
    y="y",
    text="Concept",
    title="Visualization of Definition Embeddings for INBIO Concepts",
    #labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
)

# Add hover functionality
fig.update_traces(textposition="top center", marker=dict(size=10, opacity=0.8))

# Customize layout
fig.update_layout(
    title_font_size=20,
    showlegend=False,
)

# Show interactive plot
fig.show()

# Save interactive Plotly visualization as an HTML file
fig.write_html("interactive_plot.html")

# Show interactive plot
fig.show()

