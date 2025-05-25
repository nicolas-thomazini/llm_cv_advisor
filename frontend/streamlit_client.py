import streamlit as st
import requests
import os

st.set_page_config(page_title="Analisador de Currículos com LLM", layout="centered")
st.title("📄 Analisador Inteligente de Currículos")

API_URL = os.environ.get("API_URL", "http://backend:8000")

upload_file = st.file_uploader("Envie seu currículo (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

if upload_file:
    files = {"file": (upload_file.name, upload_file.getvalue())}
    response = requests.post(f"{API_URL}/upload", files=files)

    if response.status_code == 200:
        st.success("Arquivo enviado e indexado com sucesso!")
    else:
        st.error(f"Erro no upload: {response.json().get('error')}")
    
query = st.text_input("Digite sua pergunta sobre currículo ou LinkedIn/")

if query:
    with st.spinner("Consultando modelo..."):
        response = requests.post(f"{API_URL}/ask",data={"question": query})
    
    if response.status_code == 200:
        data = response.json()
        st.markdown('### Resposta da LLM:')
        st.write(data["answer"])

        with st.expander("📚 Fontes utilizadas"):
            for src in data["sources"]:
                st.text(f"- {src}")
        
        with st.form("feedback_form"):
            feedback = st.radio("Essa resposta foi útil?", ["👍 Sim", "👎 Não"])
            submitted = st.form_submit_button("Enviar Feedback")

            if submitted:
                fb_response = requests.post(f"{API_URL}/feedback", data={"question": query, "feedback": feedback})

                if fb_response.status_code == 200:
                    st.success("Feedcback enviado com sucesso!")
                else:
                    st.error("Erro ao enviar o feedback.")