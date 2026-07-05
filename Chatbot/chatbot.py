from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# AutoTokenizer : converts text into tokens the model understands

# AutoModelForSeq2SeqLM : loads a sequence-to-sequence generation model for dialogue

model_name = "facebook/blenderbot-400M-distill"

# model is an instance of the class AutoModelForSeq2SeqLM, 
# which allows you to interact with your chosen language model.

# tokenizer is an instance of the class AutoTokenizer, 
# which optimizes your input and passes it to the language model 
# efficiently. It does so by converting your text input 
# to "tokens", which is how the model interprets the text.

# Load model (download on first run and reference local installation for subsequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Chatbot ready! (type 'exit' to quit)\n")
conversation_history = []

while True:
    #conversation_history = []
    # keep only last few exchanges (prevents confusion)
    conversation_history = conversation_history[-6:]

    

    history_string = "\n".join(conversation_history)

    # Fetch prompt from user
    input_text = input("Enter your chat> ")

    ## This will help you exit by typing exit in the prompt 
    if input_text.lower() == "exit":
        break

    prompt = history_string + f"\nUser: {input_text}\nBot:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # tokenizer(...): Converts raw text into numerical tokens the model can understand.
    # return_tensors="pt": Returns PyTorch tensors (required by the model).
    # truncation=True: Truncates input if it exceeds model limits.
    # max_length=512: Maximum number of tokens allowed as input.

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        do_sample=True,
        temperature=0.6,
        top_p=0.85
    )

    # inputs: Sends the user message and chat history to the model. This helps the chatbot understand the full conversation before replying.
    # max_new_tokens: Sets the maximum length of the reply. It stops the model from writing too much text.
    # no_repeat_ngram_size: Stops the model from repeating the same 3-word phrases again and again.
    # repetition_penalty: Reduces repeated words in the response so the output sounds more natural.
    # do_sample=True: Makes the chatbot responses more random and less fixed, so replies feel more natural.
    # temperature: Controls how creative the response is. Lower = safer answers, higher = more creative answers.
    # top_p: Keeps only the most likely word choices when generating text, which helps the response stay clear and meaningful.

    print("Output: ", outputs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # tokenizer.decode(outputs[0]): Converts the model’s output 
    # from numbers (tokens) back into readable text. 
    # The model first generates numbers, and this step turns them 
    # into a human-readable sentence.

    print("Response: ", response)

    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"Bot: {response}")


    print(conversation_history)

