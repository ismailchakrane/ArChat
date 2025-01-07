# ArChat

This application allows you to interact with Articles (PDFs).

## Setup

### Prerequisites

#### Install Ollama  

Refer to the [official Ollama documentation](https://ollama.com/) for installation instructions suitable for your operating system.

#### Download llama3.2:1b, gemma2:2b, Microsoft Phi 3 

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
conda create --name llm python=3.11.11
conda activate llm
pip install -r requirements
```

## Run

```
streamlit run app.py
```