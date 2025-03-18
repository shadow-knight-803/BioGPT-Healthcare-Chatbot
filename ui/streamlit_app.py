import sys
import os
import streamlit as st
import torch

# Ensure the project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.biogpt_model import load_biogpt
from utils.helper_functions import clear_memory, preprocess_text

# Load BioGPT Model
model, tokenizer, device = load_biogpt()

# Function to Generate Response from BioGPT
def generate_response(user_query):
    user_query = preprocess_text(user_query)
    input_ids = tokenizer.encode(user_query, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 100,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    clear_memory()
    return response

# Streamlit UI
st.title("🩺 BioGPT Healthcare Chatbot")

user_input = st.text_area("Ask a medical question:", height=100)

if st.button("Get Response"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            response = generate_response(user_input)
        st.success("✅ Response:")
        st.write(response)
    else:
        st.warning("⚠ Please enter a question before submitting.")

# Memory Usage Tab
if st.sidebar.button("Clear Memory"):
    clear_memory()
    st.sidebar.success("Memory Cleared!")
