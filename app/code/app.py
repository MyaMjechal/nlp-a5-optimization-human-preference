from flask import Flask, request, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the trained model
model = AutoModelForCausalLM.from_pretrained("myamjechal/dpo-gpt2-optimized-model")
tokenizer = AutoTokenizer.from_pretrained("myamjechal/dpo-gpt2-optimized-model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        main_response = response.split("\n\n", 1)[-1]
        return render_template("index.html", response=main_response, user_input=user_input)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
