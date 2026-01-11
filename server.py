"""
Flask server for Emotion Detection application.

Provides:
- Home route to render the UI.
- /emotionDetector route to analyze text and return formatted results.
"""

from __future__ import annotations

from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.route("/")
def home() -> str:
    """Render the home page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detector_route() -> str:
    """
    Analyze user text with Watson NLP EmotionPredict and return formatted output.

    Query parameter:
        textToAnalyze: input text to analyze.

    Returns:
        A formatted string with emotion scores and dominant emotion,
        or an error message when the input is invalid/blank.
    """
    text_to_analyze = request.args.get("textToAnalyze", default="", type=str).strip()

    result = emotion_detector(text_to_analyze)

    if result.get("dominant_emotion") is None:
        return "Invalid text! Please try again!"

    return (
        "For the given statement, the system response is "
        f"'anger': {result['anger']}, 'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, 'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )


def main() -> None:
    """Run the Flask development server on port 5000."""
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
