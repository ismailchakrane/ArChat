from langchain.llms import Ollama
from langchain.chains import LLMChain
from utils.prompts import question_answering_prompt, question_generation_prompt, training_plan_prompt, slide_generation_template

def initialize_llm_chain(model_choice):
    if model_choice == "Ollama (Llama3.2)":
        llm = Ollama(model="llama3.2:3b")
    elif model_choice == "Google Gemma2 (2B)":
        llm = Ollama(model="gemma2:2b")
    elif model_choice == "Microsoft Phi 3 Mini (3.8B)":
        llm = Ollama(model="phi3")

    qa_chain = LLMChain(llm=llm, prompt=question_answering_prompt)
    qg_chain = LLMChain(llm=llm, prompt=question_generation_prompt)
    tp_chain = LLMChain(llm=llm, prompt=training_plan_prompt)
    slide_chain = LLMChain(llm=llm, prompt=slide_generation_template)

    return qa_chain, qg_chain, tp_chain, slide_chain
