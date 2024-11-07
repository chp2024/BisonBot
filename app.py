# Import necessary libraries
from langchain_community.embeddings import HuggingFaceEmbeddings
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from openai import OpenAI as OpenAISound # Directly pull OpenAI -- is used for sound related tasks
from io import BytesIO
from chainlit.element import ElementBased
from langchain.retrievers.web_research import WebResearchRetriever
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

@cl.oauth_callback
def oauth_callback(provider_id, token, raw_user_data, default_user):
    return default_user   

# Prompt Template
prompt_template = """
You are BisonBot-- a helpful AI assistant who answers anything Howard Related. You are provided multiple context items that are related to the prompt you have to answer. 
Here is the previous conversation history with the user:

'''
{chat_history}
'''

Use this history as context to answer the user's question. Additionally, use the following pieces of context if they are relevant:

'''
{context}
'''            

Question: {question}

Think Step by step and provide an answer:
"""

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

@cl.cache
# Load OpenAI model function
def load_model():
    """
    Loads the OpenAI language model.
    """
    model = ChatOpenAI(streaming=True, temperature=0.1, model='gpt-4')  # Corrected import and usage
    return model 

@cl.cache
def load_sound_model():
    model = OpenAISound()
    return model

# Loading the local model into LLM
llm = load_model()
scraped_vector_store = Chroma(
    collection_name='howard_information',
    embedding_function=embeddings,
    persist_directory='./data/chroma'
    )
search_vector_store = Chroma(
    collection_name='howard_search_information',
    embedding_function=embeddings,
    persist_directory='./data/search_embeddings'
    )

@cl.cache
def load_search_scraper():
    search = GoogleSearchAPIWrapper()
    web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=search_vector_store, llm=llm, search=search,
    allow_dangerous_requests=True
    )
    return web_research_retriever

search_scraper = load_search_scraper()

@cl.on_chat_start
# Actions to be taken once the RAG app starts
async def factory():

    prompt = PromptTemplate(template=prompt_template,
                       input_variables=['context', 'question', 'chat_history'])  # Include chat_history
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # Key to reference the conversation history
        output_key="answer",        # Store the output
        return_messages=True        # Ensure memory returns messages for context
    )
    
    ensemble_retriever = EnsembleRetriever(
    retrievers=[
        MultiQueryRetriever.from_llm(
            retriever=search_vector_store.as_retriever(
                search_type='mmr',
                search_kwargs={'k': 3, 'fetch_k': 30, 'score_threshold': 0.9 }), llm=llm
        ),
        MultiQueryRetriever.from_llm(
            retriever=scraped_vector_store.as_retriever(
                search_type='mmr',
                serach_kwargs={
                    'k': 3,
                    'fetch_k': 30,
                    'score_threshold': 0.9
                    }),
                llm=llm
        )
        ]
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        ensemble_retriever,
        return_source_documents=True,
        memory=memory,               # Use memory for storing/referencing conversation
        verbose=False,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    
    msg = cl.Message(content="The bot is getting initialized, please wait!!!")
    await msg.send()
    msg.content = "Hi I am BisonBot. I'm here to help you with anything Howard related."
    await msg.update()
    cl.user_session.set("chain", qa_chain)

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # For now, write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)

class messageClass:
    def __init__(self,content) -> None:
        self.content = content

@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    print(cl.user_session.get("audio_mine_type"))
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    
    client = load_sound_model()
    transcription = client.audio.transcriptions.create(
        model='whisper-1',
        file=("audio.webm", audio_file, "audio/webm")
    ).text
    
    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You", 
        type="user_message",
        content=transcription,
        elements=[input_audio_el, *elements]
    ).send()
    
    await main(messageClass(transcription), audio_output=True)

# Actions to be taken once user sends a query/message
@cl.on_message
async def main(message, audio_output=False):
    chain = cl.user_session.get("chain")
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    msg = cl.Message(content="")
    try:
        search_scraper.invoke(message.content)
    except:
        print("some error while doing the search scraping occured")

    async for chunk in chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True)]),
    ):
        print(chunk)
        
        await msg.stream_token(chunk['answer'])
        if audio_output:
            client = load_sound_model()
            output_audio = client.audio.speech.create(
                model='tts-1',
                voice='nova',
                input=chunk['answer']
            ).read()
    
    if audio_output:
        output_audio_el = cl.Audio(
            name="text to speech",
            auto_play=True,  
            mime=audio_mime_type,
            content=output_audio,
        )
        answer_output_audio = await cl.Message(content="").send()
        answer_output_audio.elements = [output_audio_el]
        await answer_output_audio.update()
    await msg.send()
