from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

main = Flask(__name__)

model_name = "mrohith29/physisolver-gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@main.route("/", methods=["GET", "POST"])
def index():
    answer = None  # Initialize the answer variable

    if request.method == "POST":
        question = request.form["question"].strip()

        context = """
        Newton's first law states that an object will remain at rest or in uniform motion unless acted upon by an external force.
        Newton's second law states that force is equal to mass times acceleration (F = ma).
        Energy is the capacity to do work. The most common form of energy is mechanical energy, which includes both kinetic energy and potential energy.
        """

        inputs = tokenizer.encode_plus(
            question,
            # context,
            return_tensors="pt",  # Return PyTorch tensors
            padding="max_length",  # Pad sequences to max_length
            max_length=512,  # Limit input length
            truncation=True  # Truncate longer inputs
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

        start_token = torch.argmax(start_scores)
        end_token = torch.argmax(end_scores)

        answer_tokens = input_ids[0][start_token:end_token + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    main.run(debug=True, host="0.0.0.0", port=5000)
