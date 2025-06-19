# app.py (Updated to use deep-translator)

from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator # <-- THIS IS THE NEW LIBRARY
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the Flask app
app = Flask(__name__)

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define the main route that will serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route that will handle the analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the text from the POST request
    text_to_analyze = request.json['text']

    try:
        # 1. Translate the text to English using deep-translator
        # This is the main change. It's much simpler!
        translated_text = GoogleTranslator(source='auto', target='en').translate(text_to_analyze)

        # 2. Perform sentiment analysis on the translated text
        score = analyzer.polarity_scores(translated_text)
        sentiment_score = score['compound']

        # 3. Classify the sentiment based on the score
        if sentiment_score >= 0.05:
            sentiment_label = 'Positive'
        elif sentiment_score <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'

        # Return all the results as a JSON object
        return jsonify({
            'original_text': text_to_analyze,
            'translated_text': translated_text,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label
        })

    except Exception as e:
        # Handle potential errors (e.g., translation failed)
        print(f"An error occurred: {e}") # Printing the error helps with debugging
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)