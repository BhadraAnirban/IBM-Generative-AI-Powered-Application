pip3 install virtualenv 
virtualenv my_env # create a virtual environment my_env
source my_env/bin/activate # activate my_env

pip install transformers==4.41.2 torch==2.2.2 accelerate==0.30.1 numpy==1.26.4




Choosing the right model for your purposes is an important part of building chatbots! You can read on the different types of models available on the Hugging Face website: https://huggingface.co/models.

The Transformers library is Hugging Face's primary Python library for downloading, loading, and using pretrained models for tasks such as:

Text generation (GPT, Llama, Mistral, Gemma, etc.)
Question answering
Text classification
Summarization
Translation
Embeddings
Vision models
Speech models

Hugging Face developed and maintains the Transformers library. [en.wikipedia.org]
How it works

Models are hosted on the Hugging Face Hub.
The Transformers library downloads the model files from the Hub.
Your application uses the library to run inference locally or on a server.

How it works

Models are hosted on the Hugging Face Hub.
The Transformers library downloads the model files from the Hub.
Your application uses the library to run inference locally or on a server.

```
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

In this example:

meta-llama/Llama-3.1-8B-Instruct is a model hosted on the Hugging Face Hub.
transformers downloads and loads it.
The model itself was created by Meta, not Hugging Face.

Is Transformers only for Hugging Face models?
No.
Transformers can load models from many organizations:

Meta (Llama)
Microsoft (Phi)
Google (Gemma, T5, BERT variants)
Mistral AI
Alibaba (Qwen)
DeepSeek
Thousands of community models

Most of these models are distributed through the Hugging Face Hub, which acts as a central repository.


The Chat flow-

Now that you're all set up, let's start chatting!
There are several things you'll do to have an effective conversation with your chatbot.
Before interacting with your model, you need to initialize an object where you can store your conversation history.
1. Initialize an object to store the conversation history
Afterward, you'll do the following for each interaction with the model:
2. Encode conversation history as a string
3. Fetch prompt from user
4. Tokenize (optimize) prompt
5. Generate output from the model using prompt and history
6. Decode output
Update conversation history


chatbot_flask.py-


python3.11 -m pip install flask
python3.11 -m pip install flask_cors

Use atleast: python3.11 app.py