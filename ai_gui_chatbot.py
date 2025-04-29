import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the AI model (using GPT-like model from Hugging Face)
model_name = "microsoft/DialoGPT-small"  # You can choose a larger model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate AI response
def get_ai_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response.strip()

# Function to handle sending messages
def send_message():
    user_input = user_entry.get()
    if user_input.strip():
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, f"You: {user_input}\n", "user")
        chat_window.config(state=tk.DISABLED)

        # Get AI response
        ai_response = get_ai_response(user_input)
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, f"Bot: {ai_response}\n", "bot")
        chat_window.config(state=tk.DISABLED)

        user_entry.delete(0, tk.END)

# Create GUI
root = tk.Tk()
root.title("AI Chatbot")

# Chat window
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, height=20, width=50)
chat_window.tag_configure("user", foreground="blue")
chat_window.tag_configure("bot", foreground="green")
chat_window.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# User entry
user_entry = tk.Entry(root, width=40)
user_entry.grid(row=1, column=0, padx=10, pady=10)

# Send button
send_button = tk.Button(root, text="Send", command=send_message)
send_button.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()