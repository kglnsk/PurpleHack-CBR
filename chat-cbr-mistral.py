import streamlit as st
import requests
import json
import os 
import re
import string
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import faiss
import openai

client = openai.OpenAI(
    api_key=None,
    base_url="http://localhost:8000/v1/models",
    )

@st.cache_data
def load_faiss_index():
    with open('chunks_ref_v2.pickle', 'rb') as handle:
        chunks = pickle.load(handle)

    model = SentenceTransformer('BAAI/bge-m3',device="cpu")
    embeddings = np.load('embs-bge.npy')

    index = faiss.IndexFlatL2(embeddings.shape[1])   # build the index
    index.add(embeddings)     

    return chunks, model, index, embeddings

def get_relevant_documents(question, chunks, model, index, embeddings):
    k = 8 #NUM of retrieval candidates
    e = model.encode(question)
    dist, Idx = index.search(e.reshape(1,-1), k)
    retrievals = [chunks[i] for i in Idx.flatten()]
    return retrievals  

def generate_response(question, chunks, model, index, embeddings):
    rets = get_relevant_documents(question, chunks, model, index, embeddings)
    metadatas = list(set([list(item.metadata.keys())[0] for item in rets]))
    text = ' \n\n '.join([str(1+i) + '. ' + re.sub(r'\W+', ' ', ret.page_content) for i,ret in enumerate(rets)])
    promptstring = f"""Вы блестящий помощник, который точно отвечает на вопросы пользователей на темы связанные с документацией центрального банка России. Используя информацию, содержащуюся в нумерованных параграфах после слова ТЕКСТ, ответьте на вопрос, заданный после слова ВОПРОС. Если в тексте нет информации, необходимой для ответа, ответьте «Недостаточно информации для ответа». Структурируйте свой ответ и отвечайте на русском языке шаг за шагом.
      ТЕКСТ:
      {text}
      ВОПРОС:
      {question}"""
    print(promptstring)

    chat_completion = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[
               {"role": "user", "content": promptstring},
    ],
    temperature=0.3,
    max_tokens=2048,
)
    response = chat_completion.choices[0].message.content
    return response,metadatas,rets

def main():
    st.title("QA Банк России")

    # Initialize chat history
    if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Ready"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Введите вопрос"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            chunks, model, index, embeddings = load_faiss_index()
            response,metadatas,rets = generate_response(prompt, chunks, model, index, embeddings)
            excerpts = '\n\n'.join([str(list(rets[i].metadata.keys())[0])+' : '+re.sub(r'\W+', ' ', ret.page_content) for i,ret in enumerate(rets)])
            biblio = '\n\n'.join([f"{1+i}. {meta}" for i,meta in enumerate(metadatas)])
            full_response = f"{response}\n\n\nСПИСОК ЛИТЕРАТУРЫ:\n\n{biblio}"
            st.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()

