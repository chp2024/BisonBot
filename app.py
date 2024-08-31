# Import necessary libraries
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain_openai import ChatOpenAI
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
Use the following pieces of context to answer the question at the end.

'''
{context}
'''            

Question: {question}
"""

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

@cl.cache
# Load ChatGPT model function
def load_model():
    """
    Loads a ChatGPT language model from the specified model path.
    """
    model = ChatOpenAI(streaming=True, temperature=0)
    return(model) 

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
                       input_variables=['context', 'question'])
    
    memory = ConversationBufferMemory(
    memory_key="chat_history", output_key="answer", return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    memory=memory,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": prompt},
)
    
    msg = cl.Message(content="The bot is getting initialized, please wait!!!")
    await msg.send()
    msg.content = "Hi I am BisonBot. I'm here to help you anything Howard Related."
    await msg.update()
    cl.user_session.set("chain", qa_chain)


# Actions to be taken once user send the query/message
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
    
    