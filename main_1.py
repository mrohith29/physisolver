from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from config import MODEL_NAME

app = Flask(__name__)

model_name = "mrohith29/physisolver-gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None  # Initialize the answer variable

    if request.method == "POST":
        question = request.form["question"].strip()

        context = """
Newton's first law states that an object will remain at rest or in uniform motion unless acted upon by an external force.
Newton's second law states that force is equal to mass times acceleration (F = ma).
Newton's third law states that for every action, there is an equal and opposite reaction.
Energy is the capacity to do work. The most common form of energy is mechanical energy, which includes both kinetic energy and potential energy.
Kinetic energy is the energy of motion, given by the formula KE = 0.5 * m * v^2, where m is mass and v is velocity.
Potential energy is stored energy due to position, commonly calculated as PE = m * g * h, where h is height and g is acceleration due to gravity.
Work is defined as force applied over a distance, W = F * d * cos(θ), where θ is the angle between force and displacement.
Power is the rate at which work is done, P = W / t.
The law of conservation of energy states that energy cannot be created or destroyed, only transformed from one form to another.
Momentum is the product of mass and velocity, p = m * v, and is conserved in isolated systems.
Ohm's law in electricity states that voltage is equal to current times resistance (V = IR).
The speed of light in a vacuum is approximately 3 × 10^8 meters per second.
Waves transfer energy without transporting matter. The main properties of waves are wavelength, frequency, amplitude, and speed.
In optics, reflection and refraction describe how light behaves at boundaries between materials.
Gravitational force is the attractive force between any two masses, governed by Newton's law of universal gravitation.
Temperature is a measure of the average kinetic energy of particles. Heat is energy transferred due to a temperature difference.
Thermodynamics involves the study of heat and temperature and their relation to energy and work, with key laws like the first law (energy conservation) and the second law (entropy increases).
Electromagnetic waves include radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays, all traveling at the speed of light.
The atomic model explains the structure of atoms, with a dense nucleus containing protons and neutrons, surrounded by electrons.
Two resistors of 6Ω and 3Ω are connected in parallel. What is the equivalent resistance? is 1/R_eq = 1/R1 + 1/R2 => 1/R_eq = 1/6 + 1/3 => R_eq = 2Ω
"""



        inputs = tokenizer.encode_plus(
            question,
            context,
            return_tensors="pt",  
            padding="max_length",  
            max_length=512,  
            truncation=True  
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
    app.run(debug=True, host="0.0.0.0", port=5000)
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import MODEL_NAME

app = Flask(__name__)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Ensure pad_token is defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        question = request.form["question"].strip()
        prompt = f"Q: {question}\nA:"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "A:" in decoded:
            answer = decoded.split("A:")[-1].strip()
        else:
            answer = decoded.strip()

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
