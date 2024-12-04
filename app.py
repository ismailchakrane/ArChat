from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains.router import MultiPromptChain, LLMRouterChain
from langchain.chains.router.llm_router import RouterOutputParser
from huggingface_hub import login
import streamlit as st
import json

# Login to Hugging Face
login(token="...")

# MODEL AND PIPELINE SETUP
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1500,
    num_beams=5,
    pad_token_id=tokenizer.pad_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)

# HELPER FUNCTIONS
def truncate_context(context, question="", max_length=1024):
    """Troncature du contexte pour respecter la limite de tokens."""
    context_tokens = context.split()
    question_tokens = question.split()
    max_context_length = max_length - len(question_tokens) - 50
    if len(context_tokens) > max_context_length:
        return " ".join(context_tokens[:max_context_length])
    return context

# PROMPT TEMPLATES
question_answering_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Utilisez le contexte suivant pour répondre de manière précise et concise à la question posée.
    Si le contexte ne contient pas d'information pertinente, indiquez "Information non disponible dans le contexte".
    Contexte : {context}
    Question : {question}
    Réponse (soyez clair et direct) : """
)

question_generation_prompt = PromptTemplate(
    input_variables=["context", "num_questions"],
    template="""En fonction du contexte ci-dessous, générez {num_questions} questions claires et pertinentes,
    couvrant différents niveaux de réflexion (factuel, analytique, interprétatif).
    Variez les formulations pour encourager une réflexion approfondie.
    Contexte : {context}
    Questions : """
)

training_plan_prompt = PromptTemplate(
    input_variables=["user_answers"],
    template="""Analysez l'historique des réponses de l'utilisateur pour identifier les domaines de compétence et les lacunes.
    Proposez un plan de formation structuré sur plusieurs semaines, avec des objectifs hebdomadaires,
    des activités spécifiques et des ressources suggérées.
    Historique des réponses : {user_answers}
    Plan de formation attendu :
    - Objectif global : [Développer la compétence X]
    - Semaine 1 : [Activités, Ressources, Objectifs spécifiques]
    - Semaine 2 : [Activités, Ressources, Objectifs spécifiques]
    ..."""
)

# CHAINS SETUP
qa_chain = LLMChain(llm=llm, prompt=question_answering_prompt)
qg_chain = LLMChain(llm=llm, prompt=question_generation_prompt)
tp_chain = LLMChain(llm=llm, prompt=training_plan_prompt)

# Custom Output Parser
class JSONOutputParser(RouterOutputParser):
    """Parses output from the LLMRouterChain into the required dictionary format."""
    def parse(self, text: str) -> dict:
        try:
            # Attempt to parse as JSON
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON output: {e}. Output was: {text}")

# Router Prompt with an Attached Output Parser
router_template = """
Déterminez la tâche à effectuer en fonction de l'entrée de l'utilisateur.
Entrée : {input}
Les tâches possibles sont :
1. Répondre à une question
2. Générer des questions d'évaluation
3. Plan de formation
Retournez **uniquement** un objet JSON strictement valide avec les deux clés suivantes :
- "destination" : Une chaîne correspondant à l'une des tâches possibles.
- "next_inputs" : Un dictionnaire contenant les entrées nécessaires pour exécuter la tâche choisie.
Votre réponse (uniquement l'objet JSON valide) :
"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=JSONOutputParser()  # Attach the output parser
)

# Router Chain
router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt)

# Chain Map for MultiPromptChain
chain_map = {
    "Répondre à une question": qa_chain,
    "Générer des questions d'évaluation": qg_chain,
    "Plan de formation": tp_chain
}

multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=chain_map,
    default_chain=qa_chain
)

# STREAMLIT INTERFACE
st.title("Système Intelligent")
st.header("Consultation, Évaluation et Plan de Formation")

task_type = st.radio(
    "Choisissez une tâche :",
    ["Répondre à une question", "Générer des questions d'évaluation", "Plan de formation"]
)

if task_type == "Répondre à une question":
    question = st.text_input("Entrez votre question ici :")
    if st.button("Soumettre"):
        if question.strip():
            context = "Contexte simulé ici pour la démonstration."
            truncated_context = truncate_context(context, question)
            response = qa_chain.run(context=truncated_context, question=question)
            st.success("Réponse :")
            st.write(response)
        else:
            st.warning("Veuillez poser une question avant de soumettre.")

elif task_type == "Générer des questions d'évaluation":
    num_questions = st.number_input("Nombre de questions à générer :", min_value=1, max_value=10, value=5)
    if st.button("Générer des questions"):
        context = "Contexte simulé ici pour la démonstration."
        response = qg_chain.run(context=context, num_questions=num_questions)
        st.success("Questions générées :")
        st.write(response)

else:
    user_answers = st.text_area("Historique des réponses utilisateur :", "")
    if st.button("Générer un plan de formation"):
        if user_answers.strip():
            response = tp_chain.run(user_answers=user_answers)
            st.success("Plan de formation proposé :")
            st.write(response)
        else:
            st.warning("Veuillez fournir un historique des réponses utilisateur.")

st.sidebar.info("Système intelligent pour la consultation, l'évaluation et la formation.")
