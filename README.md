# ArChat

This application allows you to interact with Articles (PDFs) using llama3.2.

## Setup

### Prerequisites

Before you can run this project, ensure you have the `llama3.2:1b` model installed locally. To do this, follow these steps:

#### Install Ollama  

Refer to the [official Ollama documentation](https://ollama.com/) for installation instructions suitable for your operating system.

#### Download llama3.2:1b

Once Ollama is installed, you can download the `llama3.2:1b` model by executing:

```
ollama run llama3.2:1b
```

For more detailed information on the llama3.2:1b model, please consult the [official Ollama model library page](https://ollama.com/library/llama3.2:1b).

### Create and activate a Conda environment

```
conda create --name myenv
conda activate myenv
```

### Install the required packages

```
pip install -r requirements.txt
```

## Run

```
streamlit run app.py
```