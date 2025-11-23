import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import pickle as pc
import pandas as pd
import re
import json
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

st.set_page_config(page_title='Stock Predictions for Google, Apple and Microsoft',
                   initial_sidebar_state='expanded')

st.title('Stock Predictor & Insight: Google, Apple & Microsoft')

with st.sidebar:
    api_key = st.text_input('GROQ_API_KEY',type='password')

if not api_key:
    st.warning('Please insert the Groq API Key into the sidebar')
    st.stop()

@st.cache_resource
def load_assets():
    model = pc.load(open('Training/Models/model.pkl','rb'))
    scaler = pc.load(open('Training/Scaler/scaler.pkl','rb'))
    encoder = pc.load(open('Training/Encoders/encoder.pkl','rb'))

    return model, scaler, encoder

def transform_df(data, encoder, scaler):
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        df = data.to_frame().T
    else:
        df = data.copy()
    
    df1 = df.copy()
    if 'Date' in df1.columns:
        df1 = df1.drop(['Date'], axis=1, errors='ignore')
    
    df1['ticker'] = encoder.transform(df1['ticker'])
    final_df = pd.DataFrame(
        data=scaler.transform(df1), 
        columns=scaler.get_feature_names_out()
    )
    return final_df

def get_llm_chain():
    prompt = ChatPromptTemplate.from_template("""
You are a data extraction and inference assistant.

Your task is to take a natural-language sentence about a stock and convert it
into a JSON object with the following fields:

[
  "Open",
  "High",
  "Low",
  "Volume",
  "ticker",
  "Year",
  "Month",
  "Day"
]

### Rules:
- Always output **valid JSON**.
- Infer or estimate any missing values based on context.
- If the sentence gives close/high/low but not open, you may reasonably infer it from the trend.
- "Date" must be in **YYYY-MM-DD** format.
- "Year", "Month" and "Day" must match the parsed date.
- "ticker" must be ONLY: "AAPL", "MSFT", "GOOGL".
- If multiple tickers appear, choose the first one.
- All numeric values must be numbers (not strings).
- Ensure the final JSON contains **all fields with inferred values**.
- If any field is missing in the sentence, assume or estimate a reasonable value.

### User sentence and date:
{sentence} and {date}

### JSON Output:

""")
    
    llm = ChatGroq(model="llama-3.1-8b-instant")
    chain = prompt | llm
    return chain

def extract_json_from_llm_output(llm_output: str):
    match = re.search(r"\{[\s\S]*?\}", llm_output)
    if match:
        json_str = match.group()
        return json.loads(json_str)
    else:
        raise ValueError("No JSON found in LLM output")


model, scaler, encoder = load_assets()

llm_chain = get_llm_chain()


if 'history' not in st.session_state:
    st.session_state.history = []


for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if user_input:=st.chat_input():
    
    with st.chat_message('user'):
        st.markdown(user_input)
    
        st.session_state.history.append({'role':'user','content':user_input})
        

    with st.chat_message('ai'):
        with st.spinner('Thinking.....', show_time=True):
            # First LLM call to extract JSON
            response = llm_chain.invoke({'sentence': user_input, 'date': datetime.today().date()}).content
            extracted_json = extract_json_from_llm_output(response)
            transform_data = transform_df(extracted_json, encoder, scaler)
            prediction = model.predict(transform_data)[0]  # get scalar from array if needed

            st.markdown(f"**Predicted Value:** {prediction}")

            # ------------------------
            # Second LLM call: justify/critique
            # ------------------------
            critique_prompt_template = ChatPromptTemplate.from_template("""
                You are a stock analysis assistant.

                Given the user's input:
                "{user_input}"

                And the predicted stock value:
                "{prediction}"

                Provide a concise justification, reasoning, or critique of the prediction in simple terms.
                Return as plain text.
                """)
            llm = ChatGroq(model="llama-3.1-8b-instant")
            critique_chain = critique_prompt_template | llm
            critique_response = critique_chain.invoke({
                'user_input': user_input,
                'prediction': prediction
            }).content

            st.markdown(f"**Analysis / Critique:** {critique_response}")
        st.session_state.history.append({'role':'ai','content':f'Prediction: {prediction}\n Crtique: {critique_response}'})
        