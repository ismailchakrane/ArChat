# ArChat

**_ArChat: Leveraging RAG-Based Approaches for Scientific Article Querying_**

This application allows you to interact with Scientific Papers (PDFs).

## Setup

### Prerequisites

#### Install Ollama  

Refer to the [official Ollama documentation](https://ollama.com/) for installation instructions suitable for your operating system.

#### Download models

Once Ollama is installed, you can download the necessary models by executing:

```
ollama pull deepseek-r1:1.5b
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull phi3
```

### Create the environment

#### Using conda

```
conda env create -f env.yml
conda activate llm
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

## Evaluation

- **QA_evaluation.ipynb**: Evaluation of the question-answering text generation.  
- **RE_evaluation.ipynb**: Evaluation of the reference extraction.


## Demo

[Video Demonstration](https://www.youtube.com/watch?v=QOy3HRuMnvY)