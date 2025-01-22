import numpy as np
import os
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import concept_identifier
import blingfire
from load_data import load_ontology_data
from tqdm import tqdm
import abstract_concept_embedder

use_reranker = True

device = "cuda" if torch.cuda.is_available() else "cpu"

deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-base")
concept_identifier_model = concept_identifier.SpanPredModel(model).to(device)
concept_identifier_model.load_state_dict(torch.load("models/concept_identifier.pkl"))
concept_identifier_model.eval()

definition_embedding_model = AutoModel.from_pretrained("microsoft/deberta-base").to("cuda")
definition_embedding_model.load_state_dict(torch.load("models/definition_embedding_model.pkl"))
definition_embedding_model.eval()

if use_reranker:
    from concept_reranker_LLM import llama_tokenizer, target_ids, format_input
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    concept_reranker_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(device)

abstract_embedding_model = abstract_concept_embedder.TokenEmbeddingModel(AutoModel.from_pretrained("microsoft/deberta-base")).to("cuda")
abstract_embedding_model.load_state_dict(torch.load("models/abstract_token_embedding_model.pkl"))

def embed_definitions(definitions):
    input = deberta_tokenizer(definitions, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
    return definition_embedding_model(**input)["last_hidden_state"][:,0,:]

def load_ontology_embeddings():
    ontology_data = load_ontology_data()
    cache_path = "models/ontology_embeddings.pkl"
    try:
        ontology_embeddings = torch.load(cache_path)
        print(f"Loaded cached ontology embeddings from {cache_path}")
    except FileNotFoundError:
        print("No cached ontology embeddings found. Computing embeddings...")

        ontology_embeddings = {}
        with torch.no_grad():
            for concept in tqdm(ontology_data, desc="Computing concept embeddings"):
                definitions = ontology_data[concept]["definitions"] + ontology_data[concept]["generated_definitions"]
                if len(definitions) > 0:
                    embeddings = embed_definitions(definitions).mean(0)
                    ontology_embeddings[concept] = embeddings.detach()

        torch.save(ontology_embeddings, cache_path)
        print(f"Saved concept embeddings to {cache_path}")
    ontology_concept_names = list(ontology_embeddings.keys())
    ontology_embeddings = torch.stack([ontology_embeddings[x] for x in ontology_concept_names], dim=0).to("cpu")
    return ontology_concept_names, ontology_embeddings, ontology_data

ontology_concept_names, ontology_embeddings, ontology_data = load_ontology_embeddings()

def detect_concept_candidates_and_predict_embeddings(text):
    sentences = blingfire.text_to_sentences(text).split("\n")
    sentence_joined_text = " ".join(sentences)

    current_idx = 0
    sentence_spans = []
    for sentence in sentences:
        found_idx = sentence_joined_text.find(sentence, current_idx)
        end_idx = found_idx + len(sentence)
        sentence_spans.append((found_idx, end_idx, sentence))
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokens = deberta_tokenizer.tokenize(sentence)
        if current_length + len(tokens) + 2 > 500:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    chuck_starts = [sentence_joined_text.find(chunk) for chunk in chunks]

    all_detected_spans = []
    for chunk_start, chunk in zip(chuck_starts, chunks):
        input = deberta_tokenizer(chunk, return_tensors="pt", truncation=False, padding=False).to(device)
        with torch.no_grad():
            prediction = concept_identifier_model(**input)
            prediction = torch.softmax(prediction, dim=-1).cpu().numpy()[0]

            embeddings = abstract_embedding_model(**input)[0].cpu().numpy()

        discrete_prediction = np.argmax(prediction, axis=-1)
        detected_spans = []
        current_concept_span = None
        for idx in range(len(discrete_prediction)):
            if discrete_prediction[idx] == 1:
                if current_concept_span is not None:
                    detected_spans.append(current_concept_span)
                token_span = input.token_to_chars(idx+1)
                current_concept_span = [token_span.start, token_span.end, embeddings[idx]]
            elif discrete_prediction[idx] == 2:
                token_span = input.token_to_chars(idx+1)
                if current_concept_span is None:
                    current_concept_span = [token_span.start, token_span.end, embeddings[idx]]
                else:
                    current_concept_span[1] = token_span.end
            else:
                if current_concept_span is not None:
                    detected_spans.append(current_concept_span)
                    current_concept_span = None
        new_detected_spans = []
        for span in detected_spans:
            word = chunk[span[0]:span[1]]
            word = word.strip()
            new_start = chunk.find(word, span[0])
            new_detected_spans.append([new_start, new_start + len(word), span[2]])
        all_detected_spans += [[x+chunk_start, y+chunk_start, z] for x, y, z in new_detected_spans]
    #for span in all_detected_spans:
    #    print(sentence_joined_text[span[0]:span[1]])
    return sentence_joined_text, all_detected_spans, sentence_spans

def find_ontology_candidates(embedding):
    embedding = torch.tensor(embedding)
    distances = torch.cdist(embedding.reshape(1, -1), ontology_embeddings)[0]
    best_matches = torch.argsort(distances)[:8]
    return [ontology_concept_names[x.item()] for x in best_matches]

def check_definition_concept_match(ontology_candidates, concept, sentence):
    ontology_candidates = [x for x in ontology_candidates if " owl " not in x and " ontology " not in x]
    results = {}
    with torch.no_grad():
        for candidate in ontology_candidates:
            candidate_definition = ontology_data[candidate]["generated_definitions"][0]
            inputs  = format_input(candidate_definition, concept, sentence)
            inputs = llama_tokenizer([inputs], return_tensors="pt").to(device)
            probabilities = torch.softmax(concept_reranker_model(**inputs).logits[0][-2], dim=-1)

            if probabilities[target_ids[1]].item() > 0.6:
                results[candidate] = probabilities[target_ids[1]].item()
    ranking = [x[0] for x in sorted(results.items(), key=lambda x: x[1])][::-1][:5]
    return ranking

def match_spans(span_1, span_2):
    if span_1[1] <= span_2[0] or span_2[1] <= span_1[0]:
        return False
    else:
        return True

def complete_abstract_ontology_matching(abstract):
    sentence_joined_text, all_candidates, sentence_spans = detect_concept_candidates_and_predict_embeddings(abstract)
    final_candidates = []
    for candidate in all_candidates:
        candidate_string = sentence_joined_text[candidate[0]:candidate[1]]
        ontology_candidates = find_ontology_candidates(candidate[2])
        sentence = [x for x in sentence_spans if match_spans((candidate[0], candidate[1]), x[:2])][0]
        if use_reranker:
            ranking = check_definition_concept_match(ontology_candidates, candidate_string, sentence[2])
        else:
            ranking = ontology_candidates[:5]
        final_candidates.append([candidate[0], candidate[1], candidate_string, ranking])
    for x in final_candidates:
        print(x)
    quit()




abstract = '''Biological invasions have been unambiguously shown to be one of the major global causes of biodiversity loss. Despite the magnitude of this threat and recent scientific advances, this field remains a regular target of criticism - from outright deniers of the threat to scientists questioning the utility of the discipline. This unique situation, combining internal strife and an unaware society, greatly hinders the progress of invasion biology. It is crucial to identify the specificities of this discipline that lead to such difficulties. We outline here 24 specificities and problems of this discipline and categorize them into four groups: understanding, alerting, supporting, and implementing the issues associated with invasive alien species, and we offer solutions to tackle these problems and push the field forward. Invasion biology is a relatively new field, so there are ongoing debates about foundational issues regarding terminology and assessment of the causes and consequences of invasive species. These debates largely reflect differing views about the extent to which invasion biologists should advocate on behalf of native species. We surveyed reviewers of the journal Biological Invasions to obtain a better sense of how invasion biologists evaluate several foundational issues. We received 422 replies, which represented a very good response rate for an online survey of 42.5% of those contacted. Responses to several debates in the field were distributed bimodally, but respondents consistently indicated that contemporary biological invasions are unprecedented. Even still, this was not seen as justification for exaggerated language (hyperbole). In contrast to prevalent claims in the literature, only 27.3% of respondents ranked invasive species as the first or second greatest threat to biodiversity. The responses also highlighted the interaction of invasive species with other threats and the role of human activity in their spread. Finally, the respondents agreed that they need to be both more objective and better at communicating their results so that those results can be effectively integrated into management. There are many hypotheses describing the interactions involved in biological invasions, but it is largely unknown whether they are backed up by empirical evidence. This book fills that gap by developing a tool for assessing research hypotheses and applying it to twelve invasion hypotheses, using the hierarchy-of-hypotheses (HoH) approach, and mapping the connections between theory and evidence. In Part 1, an overview chapter of invasion biology is followed by an introduction to the HoH approach and short chapters by science theorists and philosophers who comment on the approach. Part 2 outlines the invasion hypotheses and their interrelationships. These include biotic resistance and island susceptibility hypotheses, disturbance hypothesis, invasional meltdown hypothesis, enemy release hypothesis, evolution of increased competitive ability and shifting defence hypotheses, tens rule, phenotypic plasticity hypothesis, Darwin's naturalization and limiting similarity hypotheses and the propagule pressure hypothesis. Part 3 provides a synthesis and suggests future directions for invasion research.'''
complete_abstract_ontology_matching(abstract)
#embedding = embed_definitions("A large black animal.")
load_ontology_embeddings()
print()