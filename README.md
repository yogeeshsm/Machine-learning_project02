
# 🤖 Offline Chatbot using ML and LLMs (Google Colab Friendly)

This project demonstrates how to build your **own offline chatbot** using advanced **Machine Learning (ML)** and **Large Language Models (LLMs)** like GPT-2. The whole setup is Colab-compatible and works even in **offline environments** (after downloading models).

---

## 🚀 Features

- Run your chatbot using Hugging Face Transformers
- Interact in a multi-turn conversation loop
- Work **offline** after setup (by saving model and tokenizer)
- Use `GPT-2` or plug in other pre-trained LLMs
- Colab-friendly with step-by-step setup

---

## 🧠 Technologies Used

- Python 🐍
- Hugging Face Transformers 🤗
- PyTorch 🔥
- Google Colab 💻

---

## 📦 Installation (Google Colab Setup)

1. **Open this notebook in Google Colab**  
   👉 [Colab Notebook Link](#) *(Replace with actual link)*

2. **Install dependencies**
   ```python
   !pip install transformers torch
   ```

3. **Load & Save Model Locally**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_name = "gpt2"  # You can change this to any other LLM available

   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)

   tokenizer.save_pretrained("./gpt2_model")
   model.save_pretrained("./gpt2_model")
   ```

4. **Use Model Offline**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

   tokenizer = AutoTokenizer.from_pretrained("./gpt2_model")
   model = AutoModelForCausalLM.from_pretrained("./gpt2_model")
   generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
   ```

---

## 💬 Sample Chatbot Loop

```python
my_name = "You"
bot_name = "Bot"
dialog = []

while True:
    user_input = input("> ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    dialog.append(f"{my_name}: {user_input}")
    prompt = "\n".join(dialog) + f"\n{bot_name}:"

    response = generator(
        prompt,
        max_length=500,
        do_sample=True,
        top_k=50,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    
    bot_reply = response[0]['generated_text'].strip()
    print(bot_reply)
    dialog.append(f"{bot_name}: {bot_reply}")
```

---

## 🔒 Offline Usage Tips

- After downloading the model/tokenizer, store it in your local project directory.
- Ensure `transformers` and `torch` are installed locally to use it offline.
- Avoid using internet-reliant tokenizers like `GPT-Neo` unless cached.

---

## 📌 Notes

- This chatbot doesn't "understand" context like ChatGPT, but simulates a realistic dialogue using LLM outputs.
- For better contextual memory, consider using embeddings or fine-tuning.

