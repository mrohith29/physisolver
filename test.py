from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model from Hugging Face Hub
model_name = "mrohith29/physisolver-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to ask a question
def ask_question(question, max_length=150):
    # Tokenize input
    inputs = tokenizer(question, return_tensors="pt").to(device)

    # Generate response
    output = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id  # Avoid warning for padding
    )

    # Decode and print response
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Model Response:\n", answer)

# Example usage
if __name__ == "__main__":
    user_query = input("Ask a physics question: ")
    ask_question(user_query)
