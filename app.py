import streamlit as st
from streamlit_tags import st_tags
import os
from prompts import *
from helpers import *
import json

def main():
    st.title("ArChat : Chat With Scientific Papers")
    st.write("This application allows you to interact with Scentific Papers using RAG-System.")

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Select an LLM model:", ["Llama3.2 (3B)", "Google Gemma2 (2B)", "Microsoft Phi 3 Mini (3.8B)"])

    # Initialize LLM based on the selected model
    if model_choice == "Llama3.2 (3B)":
        llm = OllamaLLM(model="llama3.2:3b")
    elif model_choice == "Google Gemma2 (2B)":
        llm = OllamaLLM(model="gemma2:2b")
    elif model_choice == "Microsoft Phi 3 Mini (3.8B)":
        llm = OllamaLLM(model="phi3")

    st.sidebar.header("PDF Selection")

    mapping = get_available_pdfs()
    titles = list(mapping.keys())

    search_query = st_tags(
        label="Search for an article:",
        text="Type to search...",
        suggestions=titles,
        maxtags=1,  # Only one selection allowed
        key="search_bar"
    )

    selected_pdf = None
    if search_query:
        search_query_text = search_query[0].strip()  # Remove leading/trailing spaces
        # Check for case-insensitive matching
        for title in mapping.keys():
            if search_query_text.lower() == title.lower():
                selected_pdf = mapping[title]
                break

        if selected_pdf:
            st.sidebar.success(f"Selected article: {search_query_text}")
        else:
            st.sidebar.warning("The selected article is not available. Please ensure it matches exactly.")

    # Handle file upload
    uploaded_file = st.sidebar.file_uploader("Upload a new PDF", type=["pdf"])
    if uploaded_file is not None:
        new_pdf_name = save_uploaded_file(uploaded_file)
        st.sidebar.success(f"Uploaded {new_pdf_name}. Refresh the page to see it in the search.")
        selected_pdf = os.path.join("data", new_pdf_name)

    # Display the selected PDF
    if selected_pdf:
        pdf_path = os.path.join("data", selected_pdf) if selected_pdf not in mapping else selected_pdf
        st.write(f"**Selected PDF:** {selected_pdf}")
        vectorstore = load_pdf_to_vectorstore(pdf_path)
    else:
        st.warning("No PDF selected. Please search for an article or upload a PDF.")
        
    task_type = st.radio(
        "Choose a task:",
        ["Question answering", "Summary Generation", "References Graph Generation"]
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

    elif task_type == "References Graph Generation":
        st.write("Generate a reference graph from your selected PDF.")

        if st.button("Generate Graph"):
            article_text = get_txt_content(pdf_path)

            filename_with_ext = os.path.basename(pdf_path)
            file_name_only = os.path.splitext(filename_with_ext)[0]

            reference_graph = extract_references_from_article(article_text)

            output_html = file_name_only + "_graph.html"
            html_path = reference_graph_to_html(reference_graph, out_html=output_html)

            with open(html_path, "rb") as file:
                st.download_button(
                    label="Download Graph HTML",
                    data=file,
                    file_name=output_html,
                    mime="text/html"
                )

            references_dict = reference_graph.model_dump()
            references_json = json.dumps(references_dict, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON of References",
                data=references_json,
                file_name=file_name_only + "_references.json",
                mime="application/json"
            )

            st.success("Graph generated! You can now download and open it in your browser, and also download the references JSON.")

if __name__ == "__main__":
    main()
