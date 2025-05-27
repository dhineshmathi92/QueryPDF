from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.core import get_response_synthesizer

# from llama_index.llms.openai import OpenAI
# from llama_index.llms.gemini import Gemini
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.query_engine import RetrieverQueryEngine

# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import load_index_from_storage
from llama_index.core.settings import Settings
import os
from dotenv import load_dotenv

load_dotenv("./.env")

# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
HF_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

# Load data


# embed_model = HuggingFaceEmbedding(
#     model_name="sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True
# )

embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1", trust_remote_code=True
)

Settings.embed_model = embed_model

llm = HuggingFaceInferenceAPI(
    model="mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN
)

Settings.llm = llm


def get_nodes_from_data(directory, supported_file_types=[".pdf"]):
    documents = SimpleDirectoryReader(
        input_dir=directory, required_exts=supported_file_types
    ).load_data()
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=30)

    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    return nodes


def get_vectorstore_index(directory):
    # embed_model = OpenAIEmbedding(
    #     model="text-embedding-ada-002", api_key=OPENAI_API_KEY
    # )
    # embed_model = GoogleGenAIEmbedding(
    #     model_name="text-embedding-004", api_key=GOOGLE_API_KEY
    # )
    # Initialize the embedding model

    nodes = get_nodes_from_data(directory)
    embed_dimension = 768
    faiss_index = faiss.IndexFlatL2(embed_dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes, embed_model=embed_model, storage_context=storage_context
    )
    index.storage_context.persist(persist_dir="./storage")
    return index


def get_prompt_template():

    prompt_template = """
    Context information is below.\n
    ---------------------
    {context_str}\n
    ---------------------\n
    Given the context information and not prior knowledge, answer the question: {query_str}

    """
    return PromptTemplate(template=prompt_template)


def get_query_engine(file_directory):
    if not os.path.exists("./storage"):
        # get index as retriever
        index = get_vectorstore_index(directory=file_directory)

    # # load index from disk
    vector_store = FaissVectorStore.from_persist_dir("./storage")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context=storage_context)

    # Create a retriever with similarity_top_k=5
    retriever = index.as_retriever(similarity_top_k=3)

    # initializing language model
    # llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    # llm = Gemini(
    #     model="models/gemini-1.5-flash",
    #     api_key=GOOGLE_API_KEY,
    # )

    # Initialize the response synthesizer with the custom prompt
    response_synthesizer = get_response_synthesizer(
        llm=llm, text_qa_template=get_prompt_template(), response_mode="compact"
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=response_synthesizer
    )

    return query_engine


def get_response(file_directory, user_query):

    query_engine = get_query_engine(file_directory)

    model_response = query_engine.query(user_query)

    return model_response


def main():
    file_directory = "./data"
    user_query = input("Ask anything to llm: \n")
    response = get_response(file_directory, user_query)

    print(response.response)


if __name__ == "__main__":

    main()
