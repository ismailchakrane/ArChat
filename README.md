# ArChat

This application allows you to interact with Scientific Papers (PDFs).

## Setup

### Prerequisites

#### Install Ollama  

Refer to the [official Ollama documentation](https://ollama.com/) for installation instructions suitable for your operating system.

#### Download llama3.2:3b, gemma2:2b, Microsoft Phi 3 

Once Ollama is installed, you can download the models by executing:

```
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull phi3
```

### Create the environment

#### Using conda

```
conda env create -f env_llm.yml
```

#### Using pip

```
conda create --name ArChat python=3.11.11
conda activate ArChat
pip install -r requirements.txt
```

## Run

```
streamlit run app.py
```