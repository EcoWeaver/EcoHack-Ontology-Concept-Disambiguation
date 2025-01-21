from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/send-text', methods=['POST'])
def receive_text():
    data = request.json
    entered_text = data.get('text', '')
    print(f"Received text: {entered_text}")
    return jsonify({'message': f'{entered_text}'})

@app.route('/search-concepts', methods=['POST'])
def search_concepts():
    data = request.json
    search_terms = data.get('search_terms', '')
    # Assuming the logic for searching concepts is implemented here
    # For now, let's return the search terms as dummy results
    results = search_terms.split(', ')  # Simulating search results
    return jsonify({'results': results})


if __name__ == '__main__':
    app.run(debug=True)
