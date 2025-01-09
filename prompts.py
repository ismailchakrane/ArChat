question_answering_prompt = """You are an expect in PDF scientific article analysis use the following context retrieved from the article to answer the question accurately and concisely.
    If the context does not contain relevant information, indicate "Information not available in the context."
    Context: {context}
    Question: {question}
    Answer (be clear and direct)
"""

summary_generation_template = """You are an assistant that turns text into a concise PDF summary. 
    Summarize the main points of the content below into a structured set of bullet points 
    as if creating slides for a presentation. Keep it concise and organized.

    Content:
    {context}
"""