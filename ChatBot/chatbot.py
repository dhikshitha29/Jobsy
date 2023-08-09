import streamlit as st
from streamlit_chat import message
import json
import re
import openai
from time import time,sleep
import pandas as pd
import streamlit.components.v1 as components

def chatbot():
    answers = []
    def on_input_change():
        user_input = st.session_state.user_input
        st.session_state.responses.append(user_input)
        
    def on_btn_click():
        del st.session_state['questions']
        del st.session_state['responses']

    st.session_state.setdefault('questions', [])

    st.title("CV Generator Bot")
    questions_list = [
        "What is your name?",
        "What is the name of the company you're applying for?",
        "What is the role you're applying for?",
        "How many years of experience do you have?",
        "What are your greatest professional strengths?",
        "What are your top achievements?",
        "What are your professional growth desires?",
        "What kind of work environment do you prefer?",
        "What does your ideal work day look like?",
        "Why are you excited about this particular job role?",
        "What about the job stands out to you? Why are you a great fit?"
    ]

    if 'responses' not in st.session_state.keys():
        st.session_state.questions.extend(questions_list)
        st.session_state.responses = []

    chat_placeholder = st.empty()
    st.button("Clear message", on_click=on_btn_click)

   
    answers = {}
    with st.container():
        message(st.session_state.questions[0],key=1) 
        for response, question in zip(st.session_state.responses, st.session_state.questions[1:]):
            message(response, is_user = True)
            message(question)

        for response, question in zip(st.session_state.responses, st.session_state.questions):
            answers[question] = response
    
    with st.container():
        chat_placeholder = st.empty()
        answerx = st.text_input("User Response:", on_change=on_input_change, key="user_input", value="", placeholder="Enter Text")

    with open("chatbot_results.json", "w") as f:
        json.dump(answers, f)

def details():
    # Load JSON file into a dictionary
    with open('chatbot_results.json') as f:
        data = json.load(f)

    # Create DataFrame with two columns
    df = pd.DataFrame(columns=['question', 'answer', 'label'])
    label = ["Name","Company Applying To","Role Applying To","Years of Experience","Greatest Strength","Top Achievements",
            "Professional Goal","My Ideal Work Environment","My Ideal Work Day","Why I am Excited","Why I'm a Great Fit"]

    # Populate DataFrame with JSON data
    i=0
    for key, value in data.items():
        df = df.append({'question': key, 'answer': value, 'label':label[i] }, ignore_index=True)
        i+=1
    # Print DataFrame
    # print(df)
    df.to_csv("check.csv")
    json_obj = df.to_json(orient='records')
    print(json_obj)

    json_str = json.dumps(json_obj, ensure_ascii=False)
    json_without_slash = json.loads(json_str)
    # Open a file object in write mode
    with open("details.json", "w") as file:
        file.write(json_without_slash)

    # Close the file
    # outfile.close()

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_info():
    info = open_file('details.json')
    info = json.loads(info)
    return info


openai.api_key = 'sk-jRy87Ll9VfsAAYLeObjjT3BlbkFJXb2Tyf11qPCOf5RT5M'  


def gpt3_completion(prompt,engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0, stop=['asdfasdf', 'asdasdf']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()  # force it to fix any unicode errors
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()

            filename = 'cv.txt' 
            save_file('%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def cv_creation():
    info = load_info()
    #print(info)
    text_block = ''
    for i in info:
        print("i",i)
        text_block += '%s: %s\n' % (i['label'], i['answer'])
    prompt = open_file('prompt_cover_letter.txt').replace('<<INFO>>', text_block)
    completion = gpt3_completion(prompt)
    print('\n\nCOVER LETTER:', completion)
    save_file('cover_letter.txt', completion)

     
    
def write():
    with open('cover_letter.txt', 'r') as file:
        contents = file.read()
    if st.button("Submit"):
        original_title = '<p style="font-family:Courier; color:White; font-size: 30px;">Cover Letter</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        content_style = f'<p style="font-family:Courier; color:White; font-size: 20px;">{contents}</p>'
        st.markdown(content_style, unsafe_allow_html=True)

if __name__ == "__main__":
    chatbot()
    details()
    cv_creation()
    write()