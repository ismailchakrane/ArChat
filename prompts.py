question_answering_prompt = """You are an expect in PDF scientific article analysis use the following context retrieved from the article to answer the question accurately and concisely.
    If the context does not contain relevant information, indicate "Information not available in the context."
    Context: {context}
    Question: {question}
    Answer (be clear and direct)
"""

# question_generation_prompt = """Generate exactly {num_questions} clear, concise, and relevant questions based solely on the provided context.
#     Do not generate introductions, explanations, or non-interrogative sentences.
#     Context: {context}
#     Questions:
# """

# training_plan_prompt = """Based on the following user responses, propose a structured training plan over several weeks.
#     User responses: {answers}
#     Training plan: 
# """

summary_generation_template = """You are an assistant that turns text into a concise PDF summary. 
    Summarize the main points of the content below into a structured set of bullet points 
    as if creating slides for a presentation. Keep it concise and organized.

    Content:
    {context}
"""
