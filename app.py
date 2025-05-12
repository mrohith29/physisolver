from flask import Flask, render_template, request
import google.generativeai as genai
import markdown
import os
from dotenv import load_dotenv

from config import MODEL_NAME  

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Set up Gemini
GOOGLE_API_KEY = os.getenv("gemini_api")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None

    if request.method == "POST":
        question = request.form["question"].strip()
        prompt = f"Q: {question} for this question generate response in detail in about 5-10 lines and after explanation, append the main concept and formula used as a summary in a new pargraph and highlight the formula in a Box\nA:"

        try:
            response = model.generate_content(prompt)
            markdown_text = response.text.strip()
            answer = markdown.markdown(markdown_text)
        except Exception as e:
            answer = f"<p>Error: {str(e)}</p>"

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

