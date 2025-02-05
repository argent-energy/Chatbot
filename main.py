import streamlit as st
from PIL import Image
from utils.get_usecase_map import (
    load_usecase_map,
    get_usecase_categories,
    get_usecase_functions
)
import os
import time
from dotenv import load_dotenv
#from utils.encryption_utility import decrypt_string

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
load_dotenv(".env.gpt4", override=True)
taxonomy=""

if os.environ.get("DEPLOYMENT_ENV") == "azure":
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

default_states = {
    "temperature": 0.5,
    "max_tokens": 3000,
    "access_token": "",
    "prev_usecase" : "",
    "user_authenticated": False,
    "user_email": None
}

def init_ses_states():
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def page_title_header():
    st.set_page_config(page_title="WDC Gen AI Canvas", page_icon=Image.open('images/canvas.png'))
    # top_image = Image.open('trippyPattern.png')
    # st.image(top_image)
    st.title("Gen AI Canvas")
    st.caption("Playground for exploring Use Cases")

def validate_access_token(access_token, warnings):
    global taxonomy
    try:
        decrypted_token = decrypt_string(os.environ["ENCRYPTION_KEY"], access_token)
        # Seperate the email and expiration time
        st.session_state.user_email, taxonomy_local, expires = decrypted_token.split(':')
        taxonomy = taxonomy_local

        # Convert expires to an integer
        expires = int(expires)

        # Get the current time in seconds since the epoch
        now = int(time.time())

        # Check if the token has expired
        if now > expires:
            if warnings:
                st.error("Token has expired.")
            st.session_state.user_authenticated = False
        else:
            st.session_state.user_authenticated = True
    except:
        if warnings:
            st.error("Invalid token.")
        st.session_state.user_authenticated = False

def authenticate_user():
    #if os.environ.get("ENV") == "DEV":
    return True

    if st.session_state.access_token == "":
        authentication_ui()
        return False
    else:
        validate_access_token(st.session_state.access_token, warnings=False)
        if st.session_state.user_authenticated:
            return True
        else:
            authentication_ui()
            return False

def authentication_ui():
    st.session_state.access_token = st.text_input(label="Access Token", value="", type="password")
    st.button("Authenticate", on_click=validate_access_token, args=(st.session_state.access_token, True,))

def sidebar(taxonomy):
    
    with st.sidebar:
        logo = st.image("images/logo.png")

        with st.expander(label="", expanded=True):
            st.subheader("Select the Usecase")

            menu_settings_tab, chatbot_settings_tab = st.tabs(["Menu", "Settings"])
            with chatbot_settings_tab:
                st.session_state.temperature = st.slider(
                    label="Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5
                    )
                st.session_state.max_tokens = st.slider(
                    label="Tokens",
                    min_value=100,
                    max_value=6000,
                    value=3000
                    )
            with menu_settings_tab:
                if taxonomy:
                    taxonomy_config = f"config/{taxonomy}_config.yaml"
                else:
                    taxonomy_config = f"config/{os.environ['TAXONOMY_CONFIG']}_config.yaml"
                usecase_map = load_usecase_map(taxonomy_config)
                categories = get_usecase_categories(usecase_map)

                categories.insert(0, "<Select>")
                category = st.selectbox(label="Categories", options=categories)

                usecase_functions = get_usecase_functions(usecase_map, category)
                usecases = [usecase for usecase in list(usecase_functions.keys())]
                usecases.insert(0, "<Select>")
                usecase = st.selectbox(label="Use Cases", options=usecases)

                if (st.session_state.prev_usecase != usecase and usecase != "<Select>"):
                    # print(f"New application {usecase} chosen at " + time.strftime("%H:%M:%S", time.localtime()))
                    st.session_state.prev_usecase = usecase
                    # Reset session variables here
                    for value in st.session_state:
                        # print(value)
                        if value not in default_states:
                            # print(f"   Deleting session variable : {value}")
                            del st.session_state[value]
                    # print()

                if usecase == "<Select>":
                    usecase_function = None
                else:
                    usecase_function = usecase_functions[usecase]

            if usecase_function is not None:
                return usecase_function
            else:
                return
 
# def sidebar():
    
#     categories = get_all_categories()
#     categories.insert(0, "<Select>")

#     with st.sidebar:
#         logo = st.image("logo.png")

#         # create a clickable label with text "Document Q & A" that is center aligned with
#         # st.markdown(
#         #     f"""
#         #         <div style="font-size: 20px; text-align: center; font-weight: bold;">
#         #             Case studies coming up
#         #         </div><br>
#         #         """,
#         #         unsafe_allow_html=True,
#         # ) 
        
#         with st.expander(label="Category Explorer", expanded=True):
#             menu_settings_tab, chatbot_settings_tab = st.tabs(["Use cases", "Settings"])
#             with chatbot_settings_tab:
#                 st.session_state.temperature = st.slider(
#                     label="Temperature",
#                     min_value=0.0,
#                     max_value=1.0,
#                     value=0.5
#                     )
#                 st.session_state.max_tokens = st.slider(
#                     label="Tokens",
#                     min_value=100,
#                     max_value=6000,
#                     value=3000
#                     )
#             with menu_settings_tab:
#                 category = st.selectbox(label="Categories", options=categories)

#                 # if category == "<Select>":
#                 #     st.caption(":black[Select a category above.] :point_up:")
#                 #     return

#                 category_form = check_if_category_has_form(category)
#                 if category_form:
#                     return category_form

#                 usecases = get_all_usecases(category)
#                 if not usecases:
#                     return
#                 usecases.insert(0, "<Select>")

#                 usecase = st.selectbox(label="Use Cases", options=usecases)
#                 # st.markdown(
#                 #     f"""
#                 #         <div style="font-size: 20px; text-align: center; font-weight: bold;">
#                 #             FAQ
#                 #         </div><br>
#                 #         """,
#                 #         unsafe_allow_html=True,
#                 # ) 

#                 if usecase == "<Select>":
#                     st.caption(":green[Select a use case above.] :point_up:")
#                     return

#                 usecase_form = check_if_usecase_has_form(category, usecase)
#                 if usecase_form:
#                     return usecase_form
            

if __name__ == "__main__":
    page_title_header()
    # hide_st_style = """
    #         <style>
    #         #MainMenu {visibility: hidden;}
    #         footer {visibility: hidden;}
    #         #header {visibility: hidden;}
    #         </style>
    #         """
    # st.markdown(hide_st_style, unsafe_allow_html=True)
    init_ses_states()
    if authenticate_user():
        filename = sidebar(taxonomy)
        if filename:
            exec(open(f"forms/{filename}.py").read())