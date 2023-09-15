import hashlib
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import assemblyai as aai
import langchain
import requests
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.cache import SQLiteCache
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.llms import Clarifai
from md2pdf.core import md2pdf

PROJECT_NAME = 'tube-to-text-coach'

CLF_OPENAI_USER_ID = 'openai'
CLF_CHAT_COMPLETION_APP_ID = 'chat-completion'
CLF_GPT4_MODEL_ID = 'GPT-4'
CLF_GPT35_MODEL_ID = 'GPT-3_5-turbo'

CLARIFAI_PAT = st.secrets['CLARIFAI_PAT']

aai.settings.api_key = st.secrets['ASSEMBLYAI_API_KEY']

cache_name = 'tube-to-text-cache'
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

template = """The following text delimited by three backticks is a text of a follow along "{vid_name}" 
Please split it into exercises.
Start with the name of the routine. If the name contains "follow along" words, remove it.
Then add a summary section that describes what is the routine about and who is it good for. It shouldn't be longer than a couple of sentences.
Then add a section about the equipment needed to follow the routine.
Put the number of repetitions or the time after the exercise name, don't add it to the exercise description.
If the exercise is on the right side and on the left side don't repeat it twice in the list, just put "on each side" after the number of repetitions or time.
For example
```
### Prone Pec Twist (5 repetitions on each side):  

- Begin with the right arm at 90 degrees to the side.
- Twist away, letting the foot come around for thoracic rotation.
- Repeat on the left side
```

the result should be in markdown format
If the text doesn't contain a follow-along routine that's possible to split into exercises say so
```{vid_text}```"""

prompt = PromptTemplate(template=template, input_variables=["vid_name", "vid_text"])


def generate_routine():
    # Initialize a Clarifai LLM
    clarifai_llm = Clarifai(
        pat=CLARIFAI_PAT,
        user_id=CLF_OPENAI_USER_ID,
        app_id=CLF_CHAT_COMPLETION_APP_ID,
        model_id=CLF_GPT4_MODEL_ID,
    )
    # Create LLM chain
    llm_chain = LLMChain(prompt=prompt, llm=clarifai_llm)
    with st.spinner('Generating routine'):
        result = llm_chain.run(vid_name=vid_name, vid_text=vid_text)
        # Simulate stream of response with milliseconds delay
        message_placeholder = st.empty()
        full_response = ''
        for chunk in result.splitlines():
            full_response += f"{chunk}\n"
            time.sleep(0.1)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(f"{full_response}▌")
        message_placeholder.markdown(full_response)
        # st.write(result)
        return result


@st.cache_data(show_spinner="Transcribing the video")
def transcribe_with_assembly(youtube_video_id):
    load_audio()
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(f'{st.session_state.save_dir}/{os.listdir(st.session_state.save_dir)[0]}')
    return transcript.text


@st.cache_data(show_spinner="Transcribing the video")
def transcribe_with_whisper(youtube_video_id):
    loader = GenericLoader(
        YoutubeAudioLoader([youtube_video_id], st.session_state.save_dir),
        OpenAIWhisperParser()
    )
    docs = loader.load()
    st.write(f'docs num = {len(docs)}')
    return docs[0].page_content


st.set_page_config(
    page_title="Tube-to-Text Coach",
    page_icon=":muscle:",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    with open('app_description.md') as descr:
        st.write(descr.read())

with st.container():
    left_col, right_col = st.columns(spec=[0.4, 0.6], gap='large')

    with left_col:
        youtube_link = st.text_input(value='https://www.youtube.com/watch?v=IB-g_BONpbI',
                                     label='Enter you follow-along video youtube link',
                                     help='Any valid YT URL should work')

        if youtube_link and check_video_url():
            st.video(youtube_link)

        if not check_video_url():
            st.warning('Please input a valid Youtube video link')
            st.stop()

        generate_button = st.button("Generate my routine")

    with right_col:
        if generate_button:
            vid_name = get_video_name()
            st.session_state.save_dir = f'vids/{get_hashed_name(vid_name)}'
            try:
                youtube_video_id = extract_youtube_video_id()
                vid_text = transcribe_with_assembly(youtube_video_id)
                # vid_text = transcribe_with_whisper(youtube_video_id)
                with st.expander('Text extracted from the video'):
                    st.write(vid_text)
                generated_routine = generate_routine()

                exported_pdf = tempfile.NamedTemporaryFile()
                md2pdf(pdf_file_path=exported_pdf.name,
                       md_content=generated_routine)

                with open(Path(exported_pdf.name), 'rb') as pdf_file:
                    download_pdf_button = left_col.download_button('Export to PDF', pdf_file, file_name=f'{vid_name}.pdf')
                if download_pdf_button:
                    left_col.write(f'Downloaded {vid_name}.pdf')
            except Exception as e:
                st.error(e)
            finally:
                if 'save_dir' in st.session_state:
                    remove_local_dir(st.session_state.save_dir)
