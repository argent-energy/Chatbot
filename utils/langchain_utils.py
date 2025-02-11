import os
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
load_dotenv(".env.gpt4",override=True)





def create_llm_model_argent(temperature, max_tokens, streaming=False):
    llm = AzureChatOpenAI(
        openai_api_base=os.environ["OPENAI_API_BASE_ARGENT"],
        openai_api_version=os.environ["OPENAI_API_VERSION_ARGENT"],
        deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME_ARGENT"],
        openai_api_key=os.environ["OPENAI_API_KEY_ARGENT"],
        openai_api_type=os.environ["OPENAI_API_TYPE_ARGENT"],
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )
    return llm

def create_llm_chain_argent(prompt_template, temperature, max_tokens):
    llm = create_llm_model_argent(temperature=temperature, max_tokens=max_tokens)
    return LLMChain(llm=llm, prompt=prompt_template)

