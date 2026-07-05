create a new repository on the command line
echo "# IBM-Generative-AI-Powered-Application" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/BhadraAnirban/IBM-Generative-AI-Powered-Application.git
git push -u origin main


push an existing repository from the command line
git remote add origin https://github.com/BhadraAnirban/IBM-Generative-AI-Powered-Application.git
git branch -M main
git push -u origin main



1. Gradio
Gradio is a Python library used to quickly create web-based user interfaces for Machine Learning and AI applications. It allows users to interact with models through a browser without writing HTML, CSS, or JavaScript.

In the BLIP application, Gradio provides:

An image upload component
A text output area
A local web server running on http://127.0.0.1:7860


2. Transformers
Transformers is a Hugging Face library that provides access to thousands of pretrained AI models such as:

GPT
BERT
T5
BLIP
LLaMA

In the code:
Pythonfrom transformers import BlipProcessor, BlipForConditionalGenerationShow more lines

BlipProcessor prepares the image for the model.
BlipForConditionalGeneration generates captions from the image.

The Transformers library makes it easy to use advanced AI models without training them from scratch.

3. PyTorch
PyTorch is a Deep Learning framework developed by Meta (Facebook).
It provides:

Tensors
Neural network operations
GPU acceleration
Model training and inference

In the code:
Pythonreturn_tensors="pt"Show more lines
pt stands for PyTorch.
The BLIP model internally uses PyTorch tensors and computations to generate captions.

4. Tensor
A tensor is a multi-dimensional collection of numbers.
Examples:
Scalar (0D Tensor)
Python5Show more lines
Vector (1D Tensor)
Python[1, 2, 3]Show more lines
Matrix (2D Tensor)
Python[ [1, 2], [3, 4]]Show more lines
Image Tensor
A color image is represented as:
Python[3, Height, Width]Show more lines
where:

3 = RGB channels
Height = image height
Width = image width

Neural networks can only process numbers, so all inputs (images, text, audio) are converted into tensors.

5. BLIP Image Captioning Workflow (Gradio/ImageCaptioning)
┌─────────────────────┐
│ User Uploads Image  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Gradio Interface   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      PIL Image      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   BLIP Processor    │
│ (Image → Tensor)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   PyTorch Tensor    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ BLIP Transformer    │
│       Model         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Generated Caption   │
│       Tokens        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Decode Tokens     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Caption Text     │
└─────────────────────┘





pip3 install virtualenv 
virtualenv my_env # create a virtual environment my_env
source my_env/bin/activate # activate my_env

# installing required libraries in my_env
pip install langchain==0.1.11 gradio==5.23.2 transformers==4.38.2 bs4==0.0.2 requests==2.31.0 torch==2.2.1


git remote add origin https://github.com/BhadraAnirban/IBM-Generative-AI-Powered-Application.git