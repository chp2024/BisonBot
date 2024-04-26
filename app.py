# Import necessary libraries
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import PromptTemplate

@cl.oauth_callback
def oauth_callback(provider_id, token, raw_user_data, default_user):
    return default_user   

# Prompt Template
prompt_template = """You are an helpful AI assistant and your name is BisonBot. You are kind, gentle and respectful to the user. Your job is to answer the questions related to Howard University in concise and step by step manner. 
If you don't know the answer to a question, please don't share false information.
            
Context: {context}
Question: {question}

Response for Questions asked.
answer:
"""
# embedding model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize embeddings using HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Model parameters
# path to store embeddings at vectorstore
indexpath = "data/vectorstore/"
# number of neural network layers to be transferred to be GPU for computation 
n_gpu_layers = 10
n_batch = 256

config = {'max_new_tokens': 512, 'context_length': 4096,         
            'gpu_layers': n_gpu_layers,'batch_size': n_batch,   
            'temperature': 0.1
         }


@cl.cache
# Load Llama2 model function
def load_llama2_llm():
    """
    Loads a Llama2 language model from the specified model path.

    Parameters:
    - modelpath: Path to the Llama2 language model
    Returns:
    - llm: Llama2 language model instance
    """
    model = ChatOpenAI(streaming=True, model="llama2", base_url="http://localhost:11434/v1/", temperature=0)
    return(model) 

# Loading the local model into LLM
llm = load_llama2_llm()
    
@cl.on_chat_start
# Actions to be taken once the RAG app starts
async def factory():   
    faiss_index_path = indexpath + 'temp-index'

    db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    
    prompt = PromptTemplate(template=prompt_template,
                       input_variables=['context', 'question'])
    
    # Create a retrievalQA chain using Llama2
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Replace with the actual chain type
        retriever=db.as_retriever(search_kwargs={'k': 1}),  # Assuming vectorstore is used as a retriever
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    
    msg = cl.Message(content="The bot is getting initialized, please wait!!!")
    await msg.send()
    msg.content = "Hi I am BisonBot. I'm here to help you anything Howard Related."
    await msg.update()
    cl.user_session.set("chain", chain)


# Actions to be taken once user send the query/message
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    msg = cl.Message(content="")
        
    async for chunk in chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True)]),
    ):
        print(chunk)
        await msg.stream_token(chunk['result'])

    await msg.send()
    
    