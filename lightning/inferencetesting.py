import torch
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

def load_model(checkpoint_path):
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.eval()
    return model

def chat_with_model(model, tokenizer):
    print("Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        # add a format for the model to understand
        user_input = f"Q: {user_input} A:"

        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"AI: {response}")

if __name__ == "__main__":
    testnew = False

    if testnew:
        checkpoint_path = "lightning/lightning_logs/version_3829508/checkpoints/epoch=66-step=67.ckpt"  # Replace with the path to your LLM checkpoint
        model = load_model(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        chat_with_model(model, tokenizer)
    else:
        checkpoint_path = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, cache_dir="/scratch/gilbreth/dchawra/cache/")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        chat_with_model(model, tokenizer)