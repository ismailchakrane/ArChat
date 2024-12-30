from langchain.prompts import PromptTemplate

question_answering_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following context to answer the question accurately and concisely.
    If the context does not contain relevant information, indicate "Information not available in the context."
    Context: {context}
    Question: {question}
    Answer (be clear and direct): """
)

question_generation_prompt = PromptTemplate(
    input_variables=["context", "num_questions"],
    template="""Generate exactly {num_questions} clear, concise, and relevant questions based solely on the provided context.
    Do not generate introductions, explanations, or non-interrogative sentences.
    Context: {context}
    Questions:
    """
)

training_plan_prompt = PromptTemplate(
    input_variables=["answers"],
    template="""Based on the following user responses, propose a structured training plan over several weeks.
    User responses: {answers}
    Training plan: """
)

slide_generation_template = PromptTemplate(
    input_variables=["context"],
    template="""
You are an assistant that turns text into a concise slide deck summary. 
Summarize the main points of the content below into a structured set of bullet points 
as if creating slides for a presentation. Keep it concise and organized.

Content:
{context}
"""
)
