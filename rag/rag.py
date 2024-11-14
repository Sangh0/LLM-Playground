from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_pdf_pages(file_path: str) -> PyPDFLoader:
    """Load PDF file and split it into pages.

    Args:
        - file_path (`str`): PDF file path
    """
    pdf_loader = PyPDFLoader(file_path)
    pages = pdf_loader.load_and_split()
    return pages


def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Load HuggingFace embedding model for sentence embedding.

    Args:
        - model_name (`str`): model name to load
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedding_model


def get_vectorstore(
    documents: List,
    embedding: HuggingFaceEmbeddings,
    distance_strategy: str = DistanceStrategy.COSINE,
) -> FAISS:
    """Create vector store from documents.

    Args:
        - documents (`List`): list of documents
        - embedding (`HuggingFaceEmbeddings`): sentence embedding model
        - distance_strategy (`str`): distance strategy to use, default is cosine distance, options are `cosine_distance` and `euclidean_distance`
    """
    assert distance_strategy in (
        DistanceStrategy.COSINE,
        DistanceStrategy.EUCLIDEAN_DISTANCE,
        DistanceStrategy.MAX_INNER_PRODUCT,
        DistanceStrategy.DOT_PRODUCT,
        DistanceStrategy.JACCARD,
    )
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embedding,
        distance_strategy=distance_strategy,
    )
    return vectorstore


def get_retriever(
    vectorstore: FAISS,
    search_kwargs: dict = {"k": 2, "score_threshold": 0.5},
):
    """Create retriever from vectorstore.

    Args:
        - vectorstore (`FAISS`): vectorstore object
        - search_kwargs (`dict`): search arguments for retriever
    """
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever


def generate_response(
    model_name: str,
    prompt: str,
    max_length: int = 150,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs, max_length=max_length, num_beams=5, early_stopping=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    # Load PDF file
    file_path = "/Users/hoo7311/Desktop/rag_example_at_ktds.pdf"
    pages = get_pdf_pages(file_path)

    # Load embedding model
    model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    embedding = get_embedding_model(model_name)

    # Create vector store
    documents = pages
    vectorstore = get_vectorstore(documents, embedding)

    # Create retriever
    retriever = get_retriever(vectorstore)

    # Search for similar sentences
    query = "What is the RAG model?"
    search_results = retriever.invoke(query)

    # Combine search results into a single context
    context = "\n".join(result.page_content for result in search_results)
    print("Retrieved context:")
    print(context)

    # Generate a response using the retrieved context
    generation_model_name = "gpt2"
    prompt = f"Context: {context}\nQuestion: {query}\n\nAnswer:"
    generated_response = generate_response(generation_model_name, prompt)

    print("Generated response:")
    print(generated_response)
