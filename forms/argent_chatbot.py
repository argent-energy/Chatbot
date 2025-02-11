import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from prompt_templates.argent_prompts import *
from langchain.prompts import PromptTemplate
from utils.langchain_utils import create_llm_chain_argent
from dateutil import parser
import numpy as np
from datetime import timedelta,datetime
load_dotenv(".env.gpt4",override=True)
default_values = {
    "updated_csv": "",
    "llm_output":""
}

app_to_be_migrated=''
code_to_be_converted=''


for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


def convert_seconds_to_time_format(seconds):
    time_format = str(timedelta(seconds=seconds))
    return time_format


def get_week_of_month(dt):
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(adjusted_dom / 7) + 1


    df.loc[df['a'] == a_value, 'b'].values[0]


def  time_delay_cal():
    data = {
    "IDENTIFICATIONKEY": [],
    "DELAY1": [],
    "DELAY2": [],
    "TASKNUMBER":[],
    "STATE":[],
    "UPDATEDBYDISPLAYNAME":[],
    "EXCLUDEDPARTICIPANTS":[],
    "FROMUSERDISPLAYNAME":[]
}
    task1_df = st.session_state.df[0:411]
    id_key,n=1,0
    delays2,stage_ids=[],[]
    count=1
    week_numbers=[]
    values=[]
    
    for index, row in st.session_state.df.iterrows():
        if row[0]!=id_key:
            stage_ids+=[i for i in range(1,n+1)]
            id_key=row[0]
            n=0
            count+=1
            
            
        try:
            timestamp2=row[4] # Column 4 (index 3)
            timestamp3=row[16]
            
        
        
            
            
            time2 = parser.parse(timestamp2)
            time3 = parser.parse(timestamp3)
            delay2 = (time3 - time2).total_seconds()
            delays2.append(convert_seconds_to_time_format(delay2))
            n+=1
            dt = datetime.fromisoformat(timestamp2[:-6])
            # Get the week number of the month
            week_number = get_week_of_month(dt)
            week_numbers.append(week_number)
        except:
            st.write("NAN Value is at Index: ",index)
            st.write(timestamp2)
            st.write(timestamp3)
            delays2.append(np.nan)
            n+=1
        req_id=row[0][4:19]
        values.append(st.session_state.req_df.loc[st.session_state.req_df['REQUISITION_HEADER_ID'] ==int(req_id), 'VALUE'].values[0])
        if count==20:
            break
    stage_ids+=[i for i in range(1,n+1)]
    print(len(task1_df))
    print(len(delays2))
    task1_df['TimeDelta(UPDATEDDATE-CREATEDDATE)'] = delays2
    task1_df['STAGE ID']=stage_ids
    task1_df['WEEK OF THE Month']=week_numbers
    task1_df['value']=values
    target_df= task1_df[["IDENTIFICATIONKEY","CREATOR","ASSIGNEDDATE","CREATEDDATE","FROMUSER","UPDATEDDATE",'ASSIGNEESDISPLAYNAME','TimeDelta(UPDATEDDATE-CREATEDDATE)','STAGE ID','WEEK OF THE Month','value']]
    target_df.to_csv('updated.csv', index=False)
    return target_df

def timedelay_analysis():
    template = PromptTemplate(template=TIMEDELAY_ANALYSIS_PROMPT, input_variables=["source"])
    llm_chain=create_llm_chain_argent(template,0.5,3000)
    source=pd.read_csv("updated.csv")
    llm_out=llm_chain.run(source=source)
    return llm_out
    

procurement_tab, salesbydepot_tab = st.tabs(["procurement", "Sales by Depot"])
with procurement_tab:
    with st.expander(label='please provide the files',expanded=True):
        up_load_files,paste_files=st.tabs(['Uploading of file','Pasting files'])
        with up_load_files:
            uploaded_files = st.file_uploader("Upload a CSV files", type=["csv"],accept_multiple_files=True)

            if uploaded_files:
                st.session_state.df = pd.read_csv(uploaded_files[0])
                st.session_state.req_df=pd.read_csv(uploaded_files[1])
                st.write(st.session_state.df)
                st.write(st.session_state.req_df)
                headings = st.session_state.df.columns.tolist()
                
        
            
    col1, col2 = st.columns([1,1])
    with col1:
        migration_button=st.button('Analyse the file',type='primary',use_container_width=True)
    with col2:
        clear_button=st.button('Clear',type='primary',use_container_width=True)
    if clear_button:
        st.session_state.doc = ""
    if migration_button:
            if uploaded_files is not None:
                st.session_state.updated_csv=time_delay_cal()
                st.session_state.llm_output=timedelay_analysis()

    st.write(st.session_state.updated_csv)
    st.write("\n\n\n")
    st.write(st.session_state.llm_output)
    template = PromptTemplate(template=PROCUREMENT_CHATBOT_PROMPT, input_variables=["source","user_query"])
    st.session_state.llm_chain=create_llm_chain_argent(template,0.5,3000)
    if user_query := st.chat_input("Prompt...."):

        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):

            source=pd.read_csv("updated.csv")
            
            response = st.session_state.llm_chain.run(source=source,user_query=user_query)
            st.markdown(response)
with salesbydepot_tab:
    with st.expander(label='please provide the files',expanded=True):
        up_load_files,paste_files=st.tabs(['Uploading of file','Pasting files'])
        a=0
        with up_load_files:
            uploaded_file = st.file_uploader("Upload a CSV files", type=["csv"],accept_multiple_files=False)
            

            if uploaded_file:
                
                st.session_state.salesdf = pd.read_csv(uploaded_file)
              

                st.write(st.session_state.salesdf)
                a=1
    
    queries = ['give me sales data of product ULSD from  Watling street', 'list out products bought by customer eddie stobart Ltd', 'list out products which are shipped from Truckstop depot','Give me total credit of litres to Mansfield Depot','give sales data of product Diesel High Bio-Blend (B10) in month of 08/2024']
    user_query = st.selectbox('Choose an Query:',queries )
    if user_query and a==1:

        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
        
            df_short=st.session_state.salesdf.iloc[:1000].reset_index(drop=True)
            source = df_short.astype(str).to_string()
            
            template = PromptTemplate(template=SALESBYDEPOT_CHATBOT_PROMPT, input_variables=["source"])
            
            st.session_state.salesbydepot_llm_chain=create_llm_chain_argent(template,0.5,3000)
            response = st.session_state.salesbydepot_llm_chain.run(source=source,user_query=user_query)
            st.markdown(response)
                