import streamlit as st
from prompts import *
from helpers import *

def main():
    st.title("ArChat : Chat With Scientific Papers")
    st.write("This application allows you to interact with Scentific Papers using RAG.")

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Select an LLM model:", ["Llama3.2 (3B)", "Google Gemma2 (2B)", "Microsoft Phi 3 Mini (3.8B)"])

    model_name = None
    if model_choice == "Llama3.2 (3B)":
        llm = OllamaLLM(model="llama3.2:3b")
        model_name = "llama3.2:3b"
    elif model_choice == "Google Gemma2 (2B)":
        llm = OllamaLLM(model="gemma2:2b")
        model_name = "gemma2:2b"
    elif model_choice == "Microsoft Phi 3 Mini (3.8B)":
        llm = OllamaLLM(model="phi3")
        model_name = "phi3"

    selected_pdf = None

    uploaded_file = st.file_uploader("Upload a new PDF", type=["pdf"])
    if uploaded_file is not None:
        new_pdf_name = save_uploaded_file(uploaded_file)
        selected_pdf = os.path.join("data", new_pdf_name)

    if not selected_pdf:
        st.warning("No PDF selected. Please upload your desired article PDF.")
    
    if "task_type" not in st.session_state:
        st.session_state.task_type = None

    if selected_pdf:
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Question Answering"):
                st.session_state.task_type = "Question answering"

        with col2:
            if st.button("Summary Generation"):
                st.session_state.task_type = "Summary Generation"

        with col3:
            if st.button("Reference Extraction"):
                st.session_state.task_type = "Reference Extraction"

        if st.session_state.task_type:
            st.write(f"Selected Task: {st.session_state.task_type}")

    if selected_pdf and st.session_state.task_type == "Question answering":
        question = st.text_input("Enter your question here:")
        if st.button("Submit"):
            if question.strip():
                vectorstore = load_pdf_to_vectorstore(selected_pdf)
                prompt = ChatPromptTemplate.from_template(question_answering_prompt)      
                relevent_docs = retrieve_from_vectorstore(vectorstore, query=question)
                prompt_with_context = prompt.format(context=relevent_docs, question=question)
                response = llm.invoke(prompt_with_context)

                st.success("Answer:")
                st.write(response.strip())
            else:
                st.warning("Please ask a question before submitting.")

    elif selected_pdf and st.session_state.task_type == "Summary Generation":
        if st.button("Generate Summary"):
            prompt = ChatPromptTemplate.from_template(summary_generation_template)      
            article_text = get_txt_content(selected_pdf)
            prompt_with_context = prompt.format(context=article_text)

            filename_with_ext = os.path.basename(selected_pdf)
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

    elif selected_pdf and st.session_state.task_type == "Reference Extraction":
        article_text = get_txt_content(selected_pdf)
        references = extract_references_with_prompts(article_text, model_name)
        
        if references:
            filename_with_ext = os.path.basename(selected_pdf)
            file_name_only = os.path.splitext(filename_with_ext)[0]
            main_article_title = extract_pdf_title(selected_pdf)
            output_file = save_references_graph(main_article_title, references, file_name_only)

            st.success("References extracted successfully!")
            st.subheader("References Found:")

            with open(output_file, "rb") as f:
                st.download_button(
                    label="Download References Graph",
                    data=f,
                    file_name=file_name_only + "_graph.html",
                    mime="application/html",
                )
            
            st.json(references)

        else:
            st.warning("We couldn't extract references, try again.")


if __name__ == "__main__":
    main()
