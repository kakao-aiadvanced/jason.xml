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

def get_retrival_grader() :
    ### Retrieval Grader
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm()
    retriever = get_doc_retrieval()

    system = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keywords related to the user question, grade it as relevant.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n document: {document} "),
        ]
    )

    retrieval_grader = prompt | llm | JsonOutputParser()
    question = "What is prompt?"
    docs = retriever.invoke(question)

    doc_txt = docs[0].page_content
    print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

    print( prompt )

def get_joke() :

    llm = get_llm()
    # retriever = get_doc_retrieval()

    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate

    joke_query = "Tell me a joke."

    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        {format_instructions}
        {query}
        """,

        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    result = chain.invoke({"query": joke_query})

    print( result )





if __name__ == '__main__':
    # doc_retrieval("What is Task Decomposition?")
    #
    # get_retrieval()

    # get_retrival_grader()


    # get_joke()
    pass