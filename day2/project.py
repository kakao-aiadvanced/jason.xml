import getpass
import os
# os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# https://velog.io/@kwon0koang/%EB%A1%9C%EC%BB%AC%EC%97%90%EC%84%9C-Llama3-%EB%8F%8C%EB%A6%AC%EA%B8%B0

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
os.environ["LANGCHAIN_PROJECT"] = "My_project"

def get_llm() :
    llm = ChatOpenAI(model="gpt-4o-mini")
    return llm

def get_doc_retrieval() :
    # load
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=( urls ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    docs = loader.load()

    # print( docs )
    #
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    #
    # print( splits )
    # text-embedding-3-small
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents( documents=splits, embedding=embeddings )
    #
    # # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever( search_type='similarity', search_kwargs={'k': 6} )

    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_doc_list(docs):
    idx = 1
    result = ""
    for doc in docs :
        result = result + ( f"DOCUMENT {str(idx)} : {doc.page_content}\n\n" )
        idx += 1
    return result


def get_retrieval() :
    llm = get_llm()
    retriever = get_doc_retrieval()
    prompt = hub.pull("rlm/rag-prompt")


    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print( prompt )
    print( {"context": retriever | format_docs, "question": RunnablePassthrough()} )
    # print( {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt )

    result = rag_chain.invoke("What is Sensory Memory?")

    print( result )


def get_retrieval_answer() :
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm()
    retriever = get_doc_retrieval()

    system = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n context: {context} "),
        ]
    )

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    question = "What is prompt?"
    docs = retriever.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})
    print(generation)


def get_retrival_grader() :
    ### Retrieval Grader
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm()
    retriever = get_doc_retrieval()

    system = """You are a grader assessing relevance of a retrieved documents to a user question.
        If each document contains keywords related to the user question, grade it as relevant.
        It does not need to be a stringent test. 
        The goal is to filter out erroneous retrievals.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        Score must be provided each documents.
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n documents: \n\n{document} "),
        ]
    )

    retrieval_grader = prompt | llm | JsonOutputParser()
    question = "What is prompt?"
    docs = retriever.invoke(question)

    doc_txt = docs[0].page_content
    print(retrieval_grader.invoke({"question": question, "document": format_doc_list( docs )}))


def get_retrieval_hallucination_grader() :
    ### Hallucination Grader
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm()
    retriever = get_doc_retrieval()

    question = "What is prompt?"
    docs = retriever.invoke(question)

    system = """You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
        Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. 
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "documents: {documents}\n\n answer: {generation} "),
        ]
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    hallucination_grader.invoke({"documents": docs, "generation": generation})


if __name__ == '__main__':
    # doc_retrieval("What is Task Decomposition?")
    #
    # get_retrieval()
    get_retrieval_answer()

    # get_retrival_grader()


    # get_joke()

    # retriever = get_doc_retrieval()
    #
    # question = "What is prompt?"
    # docs = retriever.invoke(question)
    # print( len( docs ) )
    # print( format_doc_list( docs ) )
    #
    #

