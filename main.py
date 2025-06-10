import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize models
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite") # Changed to a generally available model

# Load FAISS vector database
db_path = "faiss_index"
vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Aeon, an AI assistant for NCS Soft Solution bank audit software company. "
            "Your primary objectives are to:\n"
            "1. Provide accurate and concise information about our bank audit software\n"
            "2. Answer questions strictly based on the provided context\n"
            "3. Maintain a professional and helpful tone\n"
            "4. If the answer is not in the context, clearly state 'I don't have enough information to answer that query'\n"
            "5. Prioritize context-based information over general knowledge\n"
            "\nKey focus areas: software capabilities, audit processes, compliance solutions, and company information"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Respond professionally and directly, using only the information provided in the context."
        ),
    ]
)


# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the RAG chain
rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: x["question"]) | retriever | format_docs
    )
    | prompt
    | llm
)

# Chatbot interaction loop
def chatbot_qa(question: str, chat_history: list):
    response = rag_chain.invoke({"question": question, "chat_history": chat_history})
    return response.content

if __name__ == "__main__":
    chat_history = []
    print("Chatbot initialized. Type 'exit' to end the conversation.")

    while True:
        user_question = input("You: ")
        if user_question.lower() == 'exit':
            break

        ai_response = chatbot_qa(user_question, chat_history)
        print(f"AI: {ai_response}")

        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=ai_response))
