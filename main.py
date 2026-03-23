import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

Data_path= "data"
Persist_dir = "./.chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


if not os.path.exists(Persist_dir):
    print("Creating vector store for the first time...")

    loader = PyPDFDirectoryLoader(Data_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=Persist_dir
    )

    print("Vector store created successfully.")

else:
    print("Loading existing vector store...")
    vectorstore = Chroma(
        persist_directory=Persist_dir,
        embedding_function=embeddings
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.3
)


template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. Answer only from the provided context. "
        "If the answer is not present in the context, say: "
        "'I don't know based on the provided documents.' "
        "Also, keep the answer clear and concise."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])

while True:
    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        print("\nAnswer:")
        print("I don't know based on the provided documents.")
        continue

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = template.format_messages(
        context=context,
        question=query
    )

    result = llm.invoke(prompt)

    print("\nAnswer:")
    print(result.content)

    print("\nSources:")
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        print(f"{i}. {source} | page {page}")