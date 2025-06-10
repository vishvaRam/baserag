import json
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv  # Import load_dotenv


def create_and_store_embeddings(json_file_path, db_save_path="faiss_index"):
    """
    Creates document embeddings from a JSON file and stores them in a FAISS vector store.

    Args:
        json_file_path (str): The path to the input JSON file.
        db_save_path (str): The directory to save the FAISS index. Defaults to "faiss_index".
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file_path}'. "
              "Please ensure it's a valid JSON file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    texts = list(data.values())

    try:
        # Load API key from environment variables using dotenv
        # GoogleGenerativeAIEmbeddings automatically looks for GOOGLE_API_KEY
        # in environment variables.
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        print(f"Error initializing GoogleGenerativeAIEmbeddings: {e}")
        print("Please ensure GOOGLE_API_KEY is set in your environment variables or .env file.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    try:
        docs = text_splitter.create_documents(texts)
    except Exception as e:
        print(f"Error creating documents with text splitter: {e}")
        return

    try:
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(db_save_path)
        print(f"Embeddings successfully created and saved to '{db_save_path}'.")
    except Exception as e:
        print(f"Error creating or saving the FAISS vector store: {e}")


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    json_file = "Data/questions.json"

    # Ensure GOOGLE_API_KEY is available
    if os.getenv("GOOGLE_API_KEY") is None:
        print("\nWarning: GOOGLE_API_KEY environment variable is not set.")
        print("Please create a .env file in the same directory as this script with content like:")
        print("GOOGLE_API_KEY='your_google_api_key_here'")
        print("Or set it as an environment variable before running the script.")

    create_and_store_embeddings(json_file)
