import streamlit as st
import os
from prompts import *
from helpers import *

def main():
    st.title("ArChat : Chat With Scientific Papers")
    st.write("This application allows you to interact with Scentific Papers using RAG-System.")

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Select an LLM model:", ["Ollama (Llama3.2)", "Google Gemma2 (2B)", "Microsoft Phi 3 Mini (3.8B)"])

    # Initialize LLM based on the selected model
    if model_choice == "Ollama (Llama3.2)":
        llm = OllamaLLM(model="llama3.2:3b")
    elif model_choice == "Google Gemma2 (2B)":
        llm = OllamaLLM(model="gemma2:2b")
    elif model_choice == "Microsoft Phi 3 Mini (3.8B)":
        llm = OllamaLLM(model="phi3")

    st.sidebar.header("PDF Selection")

    mapping = get_available_pdfs()
    titles = list(mapping.keys())
    selected_title = st.sidebar.selectbox("Select an article:", titles)
    selected_pdf = mapping[selected_title]

    uploaded_file = st.sidebar.file_uploader("Upload a new PDF", type=["pdf"])
    if uploaded_file is not None:
        new_pdf_name = save_uploaded_file(uploaded_file)
        st.sidebar.success(f"Uploaded {new_pdf_name}. Refresh the dropdown to see it.")
        selected_pdf = new_pdf_name

    pdf_path = os.path.join("data", selected_pdf)

    st.write(f"**Selected PDF:** {selected_pdf}")
    vectorstore = load_pdf_to_vectorstore(pdf_path)

    task_type = st.radio(
        "Choose a task:",
        ["Question answering", "Summary Generation", "Graph Generation"]
    )

    if task_type == "Question answering":
        input_method = st.radio("Choose input method", ["Text Q&A", "Audio Q&A"])

        if input_method == "Text Q&A":
            question = st.text_input("Enter your question here:")
            if st.button("Submit"):
                if question.strip():
                    prompt = ChatPromptTemplate.from_template(question_answering_prompt)      
                    relevent_docs = retrieve_from_vectorstore(vectorstore, query=question)
                    prompt_with_context = prompt.format(context=relevent_docs, question=question)
                    response = llm.invoke(prompt_with_context)

                    st.success("Answer:")
                    st.write(response.strip())
                else:
                    st.warning("Please ask a question before submitting.")

        elif input_method == "Audio Q&A":
            pass
            # if st.button("Ask by Audio"):
            #     question = recognize_audio()  # Capture and recognize the audio question
            #     if question:
            #         response = qa_chain.run(context=context, question=question)
            #         st.success("Answer:")
            #         st.write(response.strip())

    elif task_type == "Summary Generation":
        if st.button("Generate Summary"):
            prompt = ChatPromptTemplate.from_template(summary_generation_template)      
            article_text = get_txt_content(pdf_path)
            prompt_with_context = prompt.format(context=article_text)

            filename_with_ext = os.path.basename(pdf_path)
            file_name_only = os.path.splitext(filename_with_ext)[0]
            output_pdf = file_name_only + "_summary.pdf"

            response = llm.invoke(prompt_with_context)

            markdown_to_pdf(response, output_file=output_pdf)
        
            full_pdf_path = os.path.join("summary", output_pdf)

            with open(full_pdf_path, "rb") as file:
                btn = st.download_button(
                    label="Download Summary PDF",
                    data=file,
                    file_name=output_pdf,
                    mime="text/pdf"
                )
            st.success("Summary generated! You can now download and open it in your browser.")

    elif task_type == "Graph Generation":
        st.write("Generate a concept graph from your selected PDF.")

        if st.button("Generate Graph"):
            article_text = get_txt_content(pdf_path)
            concept_graph = extract_concepts_from_article(article_text)

            filename_with_ext = os.path.basename(pdf_path)
            file_name_only = os.path.splitext(filename_with_ext)[0]

            output_html = file_name_only + "_graph.html"
            show_concept_graph_in_notebook(concept_graph, out_html=output_html)

            full_html_path = os.path.join("graphs", output_html)

            with open(full_html_path, "rb") as file:
                btn = st.download_button(
                    label="Download Graph HTML",
                    data=file,
                    file_name=output_html,
                    mime="text/html"
                )
            st.success("Graph generated! You can now download and open it in your browser.")

if __name__ == "__main__":
    main()
