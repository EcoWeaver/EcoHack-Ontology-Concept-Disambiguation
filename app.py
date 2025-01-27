from flask import Flask, request, jsonify
from flask_cors import CORS
from text_processing import *
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    entered_text = data.get('text', '')
    final_candidates, sentence_joined_text = complete_abstract_ontology_matching(entered_text)

    # Prepare HTML with spans having unique IDs
    highlighted_text = sentence_joined_text
    concept_info = []
    for i, candidate in enumerate(sorted(final_candidates, key=lambda x: -x[0])):  # From end to start to avoid index shifts
        start, end, candidate_string, ranking = candidate
        span_id = f"concept-{i}"  # Unique ID for each highlighted concept
        span_html = f'<span id="{span_id}" class="highlight" onclick="showMatchedConcepts(\'{span_id}\')" title="Click to view matches">{candidate_string}</span>'
        highlighted_text = (
            highlighted_text[:start] + span_html + highlighted_text[end:]
        )

        # Collect information for the frontend, including scores
        matched_concepts = [
            {
                'concept': concept,
                'score': score,
                'definition': load_ontology_definition(concept)
            } for concept, score in ranking
        ]
        concept_info.append({
            'id': span_id,
            'original_text': candidate_string,
            'matched_concepts': matched_concepts
        })

    return jsonify({
        'modified_text': highlighted_text,
        'concept_info': concept_info  # Send detailed info about each concept, including scores
    })


@app.route('/search-related-abstracts', methods=['POST'])
def search_related_abstracts_app():
    data = request.json
    entered_text = data.get('text', '')
    related_texts_with_concepts = search_related_abstracts(entered_text)  # Returns [(text, concepts), ...]

    # Convert each text-concepts pair into highlighted HTML
    highlighted_abstracts = []
    for text, concepts in related_texts_with_concepts:
        # Sort concepts by length (longest first) to handle overlapping concepts
        sorted_concepts = sorted(concepts, key=len, reverse=True)

        # Initialize markers array to track which parts of text are already highlighted
        text_length = len(text)
        highlighted = [False] * text_length

        # First pass: mark all matches in the highlighted array
        matches = []  # Store all matches to process later
        for concept in sorted_concepts:
            start = 0
            while True:
                index = text.lower().find(concept.lower(), start)
                if index == -1:
                    break

                # Check if any part of this match is already highlighted
                if not any(highlighted[index:index + len(concept)]):
                    matches.append((index, index + len(concept), concept))
                    # Mark this region as highlighted
                    for i in range(index, index + len(concept)):
                        highlighted[i] = True

                start = index + 1

        # Sort matches by position (from end to start to preserve indices)
        matches.sort(key=lambda x: x[0], reverse=True)

        # Second pass: apply the highlights
        result = text
        for start, end, concept in matches:
            actual_text = result[start:end]
            span_html = f'<span class="highlight" title="{concept}">{actual_text}</span>'
            result = result[:start] + span_html + result[end:]

        highlighted_abstracts.append(result)

    return jsonify({
        'related_abstracts': highlighted_abstracts
    })

@app.route('/search-concepts', methods=['POST'])
def search_concepts():
    data = request.json
    search_terms = data.get('search_terms', '')
    # Assuming the logic for searching concepts is implemented here
    # For now, let's return the search terms as dummy results
    results = search_terms.split(', ')  # Simulating search results
    return jsonify({'results': results})


if __name__ == '__main__':
    app.run(debug=False)
