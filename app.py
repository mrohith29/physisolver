from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from config import MODEL_NAME

# Initialize Flask application
app = Flask(__name__)

model_name = "mrohith29/physisolver-gpt2"

# Load tokenizer and model for question answering
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# Ensure that the pad_token is set (required for model padding)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None  # Initialize the answer variable

    if request.method == "POST":
        # Retrieve the user's question from the form input
        question = request.form["question"].strip()

        # Simple physics context for the QA model
        context = """
        Newton's first law states that an object will remain at rest or in uniform motion unless acted upon by an external force.
        Newton's second law states that force is equal to mass times acceleration (F = ma).
        The speed of light in a vacuum is approximately 299,792 kilometers per second.
        Energy is the capacity to do work. The most common form of energy is mechanical energy, which includes both kinetic energy and potential energy.
        """

        # Tokenize the input question and context
        inputs = tokenizer.encode_plus(
            question,
            context,
            return_tensors="pt",  # Return PyTorch tensors
            padding="max_length",  # Pad sequences to max_length
            max_length=512,  # Limit input length
            truncation=True  # Truncate longer inputs
        )

        # Extract input_ids and attention_mask
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Forward pass through the model to get the start and end positions of the answer
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

        # Get the most likely start and end tokens for the answer
        start_token = torch.argmax(start_scores)
        end_token = torch.argmax(end_scores)

        # Decode the answer from the tokens
        answer_tokens = input_ids[0][start_token:end_token + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Render the result in the template (index.html)
    return render_template("index.html", answer=answer)

# Run the app in debug mode
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
