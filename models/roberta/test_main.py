
import onnxruntime as ort
import numpy as np
from transformers import RobertaTokenizer

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
ort_session = ort.InferenceSession('/home/solvz/Hackathon/iris/models/roberta/model.onnx')

# Example text
query = "What is the capital of France?"
context = "France is a country in Europe. The capital of France is Paris."

# Manually tokenize the input
encoded_inputs = tokenizer(
    query,
    context,
    return_tensors="np",
    padding=True,
    truncation=True
)

# Convert inputs to numpy arrays
input_dict = {k: np.array(v) for k, v in encoded_inputs.items()}

# Run inference
outputs = ort_session.run(None, input_dict)

# Assuming the first element of outputs is the logits or token IDs
token_ids = np.argmax(outputs[0], axis=-1)

# Decode the token IDs to get the output text
decoded_output = tokenizer.decode(token_ids[0], skip_special_tokens=True)

# Save the decoded output to a text file
with open('output.txt', 'w') as f:
    f.write(decoded_output)

print("Output saved to output.txt")