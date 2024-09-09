# Import necessary libraries
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.llms import OpenAI  # Correct import
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

@cl.oauth_callback
def oauth_callback(provider_id, token, raw_user_data, default_user):
    return default_user   

# Prompt Template
prompt_template = """
You are a helpful AI assistant and your name is BisonBot. You are provided multiple context items that are related to the prompt you have to answer. 
Here is the previous conversation history with the user:

'''
{chat_history}
'''

Use this history as context to answer the user's question. Additionally, use the following pieces of context if they are relevant:

'''
{context}
'''            

Question: {question}
"""

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

@cl.cache
# Load OpenAI model function
def load_model():
    """
    Loads the OpenAI language model.
    """
    model = OpenAI(streaming=True, temperature=0)  # Corrected import and usage
    return model 

# Loading the local model into LLM
llm = load_model()
    
@cl.on_chat_start
# Actions to be taken once the RAG app starts
async def factory():   

    vector_store = Chroma(
    collection_name='howard_information',
    embedding_function=embeddings,
    persist_directory='./data/chroma'
    )
    
    prompt = PromptTemplate(template=prompt_template,
                       input_variables=['context', 'question', 'chat_history'])  # Include chat_history
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # Key to reference the conversation history
        output_key="answer",        # Store the output
        return_messages=True        # Ensure memory returns messages for context
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vector_store.as_retriever(search_kwargs={"k": 2}),
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


# Actions to be taken once user sends a query/message
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    msg = cl.Message(content="")
        
    async for chunk in chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True)]),
    ):
        print(chunk)
        await msg.stream_token(chunk['answer'])

    await msg.send()
