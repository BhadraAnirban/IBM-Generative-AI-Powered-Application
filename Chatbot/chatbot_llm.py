from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

# warnings.filterwarnings("ignore") suppresses unnecessary 
# Hugging Face warning messages to keep the output cleaner.
warnings.filterwarnings("ignore")

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.unk_token

# pad_token: In transformer models, inputs in a batch must often be the same length. Shorter sequences are padded with a special token called the padding token (pad_token). This tells the model which parts of the input are real words and which are filler.
# device_map: Controls where the model runs (e.g., CPU or GPU) and ensures it is correctly loaded on the available device.
# torch_dtype: Sets the numerical precision of computations (e.g., float32 or float16) to balance speed, memory usage, and accuracy.

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype=torch.float32
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Give short and concise answers in 2-3 lines."
    }
]

# messages: This is the full conversation history between the user and the AI. Each message has two parts: role who is speaking and content what they are saying

# There are three types of roles in a chat-based AI system. The system role defines the rules and behavior of the AI, such as how it should respond. The user role represents the questions or inputs given by the person using the chatbot. The assistant role contains the AI’s responses generated based on both the system instructions and user input, forming the conversation flow.

print("Chatbot started. Type 'exit' to quit.\n")
while True:
    user_input = input("> ")

    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})
    messages = [messages[0]] + messages[-10:]
    tokenized = tokenizer.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=True,
      return_tensors="pt",
      return_dict=True,
      max_length=512
    )

    with torch.inference_mode():
      outputs = model.generate(
          tokenized["input_ids"],
          attention_mask=tokenized["attention_mask"],
          max_new_tokens=60,
          temperature=0.5,
          top_p=0.8,
          do_sample=True,
          repetition_penalty=1.3,
          no_repeat_ngram_size=3,
          pad_token_id=tokenizer.pad_token_id
      )

    response = tokenizer.decode(
        outputs[0][tokenized["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    print(f"Bot: {response}\n")
    messages.append({"role": "assistant", "content": response})


