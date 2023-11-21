import hashlib
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import langchain
import requests
import streamlit as st
from langchain import hub
from langchain.cache import SQLiteCache
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from md2pdf.core import md2pdf

PROJECT_NAME = 'tube-to-text-coach'

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


def check_video_url():
    checker_url = f"https://www.youtube.com/oembed?url={youtube_link}"
    response = requests.get(checker_url)
    return response.status_code == 200


def extract_youtube_video_id():
    # Regular expression to match YouTube video IDs
    pattern = r"((?<=(v|V)/)|(?<=be/)|(?<=(\?|\&)v=)|(?<=embed/))([\w-]+)"
    return match.group() if (match := re.search(pattern, youtube_link)) else None


def get_video_name():
    checker_url = f"https://www.youtube.com/oembed?url={youtube_link}"
    response = requests.get(checker_url)
    return response.json()['title']


def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def log_to_langsmith():
    os.environ['LANGCHAIN_API_KEY'] = st.secrets['LANGCHAIN_API_KEY']
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = PROJECT_NAME


def remove_local_dir(local_dir_path):
    print(f'Removing local files in {local_dir_path}')
    shutil.rmtree(local_dir_path, ignore_errors=True)


def load_audio():
    loader = YoutubeAudioLoader([youtube_link], st.session_state.save_dir)
    list(loader.yield_blobs())


log_to_langsmith()

routine_extractor_prompt = hub.pull("aaalexlit/sport-routine-to-program")
routine_extractor_prompt_short = hub.pull("aaalexlit/sport-routine-to-program-short")


def generate_routine():
    # Initialize the LLM
    llm = ChatOpenAI(model='gpt-4-1106-preview',
                     temperature=0.4,
                     openai_api_key=st.session_state.openai_api_key)
    # Create LLM chain
    if st.session_state.short_version:
        prompt = routine_extractor_prompt_short
    else:
        prompt = routine_extractor_prompt
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain.run(vid_name=vid_name, vid_text=vid_text)


def simulate_steam_response(result):
    # Simulate stream of response with milliseconds delay
    message_placeholder = st.empty()
    full_response = ''
    for chunk in result.splitlines():
        full_response += f"{chunk}\n"
        time.sleep(0.1)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(f"{full_response}▌")
    message_placeholder.markdown(full_response)


def remove_openapi_key():
    if 'openai_api_key' in st.session_state:
        del st.session_state['openai_api_key']


@st.cache_data(show_spinner="Transcribing the video")
def transcribe_with_whisper(youtube_video_id):
    loader = GenericLoader(
        YoutubeAudioLoader([youtube_video_id], st.session_state.save_dir),
        OpenAIWhisperParser(api_key=st.session_state.openai_api_key)
    )
    docs = loader.load()
    return docs[0].page_content


def show_transcript():
    if show_extracted_text:
        with st.expander('Video transcript'):
            st.write(vid_text)


@st.cache_data
def load_transcript():
    with open('transcript.txt') as transcript_file:
        return transcript_file.read()


@st.cache_data
def load_transcript():
    with open('transcript.txt') as transcript_file:
        return transcript_file.read()


@st.cache_data
def load_routine(short_version: bool):
    filename = 'short_routine.md' if short_version else 'routine.md'
    with open(filename) as f:
        return f.read()


def export_to_pdf():
    exported_pdf = tempfile.NamedTemporaryFile()
    md2pdf(pdf_file_path=exported_pdf.name,
           md_content=generated_routine)
    postfix = 'short' if st.session_state.short_version else ''
    full_pdf_name = f'{vid_name} {postfix}.pdf'
    with open(Path(exported_pdf.name), 'rb') as pdf_file:
        export_pdf_button = left_col.download_button('Export to PDF', pdf_file,
                                                       file_name=full_pdf_name)
    if export_pdf_button:
        left_col.write(f'Downloaded {full_pdf_name}')
        st.stop()


def manage_openai_api_key():
    if openai_api_key := st.text_input(label='OpenAI API key', type='password'):
        if openai_api_key.startswith('sk-'):
            st.session_state.openai_api_key = openai_api_key
        elif not openai_api_key.startswith('sk-'):
            st.write("This doesn't look like a valid openai API key")
            remove_openapi_key()
    else:
        remove_openapi_key()


st.set_page_config(
    page_title="Tube-to-Text Coach",
    page_icon=":muscle:",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    manage_openai_api_key()
    with open('app_description.md') as descr:
        st.write(descr.read())
    st.subheader('**Demo**')
    st.video('https://www.youtube.com/watch?v=rj76EDbaOX4')

# with st.columns(spec=[0.4, 0.6], gap='large')[0]:

with st.container():
    left_col, right_col = st.columns(spec=[0.4, 0.6], gap='large')
    no_api_key = 'openai_api_key' not in st.session_state

    with left_col:
        with st.form('options'):
            st.subheader('Enter your follow-along video youtube link')
            label_visibility = 'visible' if no_api_key else 'collapsed'
            youtube_link = st.text_input(value='https://www.youtube.com/watch?v=1a7URy4pLfw',
                                         label="Without entering the OpenAI API key, it is only possible to view the "
                                               "pre-loaded output",
                                         help='Any valid YT URL should work',
                                         disabled=no_api_key,
                                         label_visibility=label_visibility)

            if check_video_url():
                st.video(youtube_link)

            show_extracted_text = st.checkbox('Show video transcript', value=False)
            short_version = st.checkbox('Short version',
                                        help="Just list the exercises without any description",
                                        key='short_version',
                                        value=False)

            generate_button = st.form_submit_button("Generate my routine")

            if not check_video_url():
                st.warning('Please input a valid Youtube video link')
                st.stop()

    with right_col:
        if generate_button:
            if no_api_key:
                vid_name = '5 Minute Morning Mobility V2'
                vid_text = load_transcript()
                show_transcript()
                generated_routine = load_routine(st.session_state.short_version)
            else:
                vid_name = get_video_name()
                st.session_state.save_dir = f'vids/{get_hashed_name(vid_name)}'
                try:
                    youtube_video_id = extract_youtube_video_id()
                    vid_text = transcribe_with_whisper(youtube_video_id)
                    show_transcript()

                    generated_routine = generate_routine()

                except Exception as e:
                    st.error(e)
                finally:
                    if 'save_dir' in st.session_state:
                        remove_local_dir(st.session_state.save_dir)
            if generated_routine:
                with st.spinner('Generating routine'):
                    simulate_steam_response(generated_routine)
                export_to_pdf()
