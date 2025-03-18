import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load BioGPT Model
def load_biogpt():
    model_name = "microsoft/BioGPT-Large"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)

    return model, tokenizer, device
