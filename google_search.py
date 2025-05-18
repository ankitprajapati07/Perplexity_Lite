from typing import Literal, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from faiss_vectordb import store_message, retrieve_relevant_messages

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")

# Initialize MemorySaver
memory = MemorySaver()


# Define chatbot state
class State(MessagesState):
    summary: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, streaming=True)


# Google Search Function
def google_search(query: str) -> List[str]:
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_SEARCH_API_KEY, "cx": GOOGLE_SEARCH_CX, "q": query}
    response = requests.get(search_url, params=params)
    results = response.json().get('items', [])
    return [item['link'] for item in results[:7]]  # Return top 3 links


# Web Scraping Function
def scrape_webpage(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.get_text(separator=" ", strip=True)  # Extract all visible text
        return content[:10000]  # Limit to 10,000 characters
    except Exception:
        return ""


# Determine if Google Search is needed
def should_use_google_search(query: str) -> bool:
    keywords = ["news", "latest", "trending", "stock", "company", "real-time", "today", "update", "any year mentioned"]
    return any(keyword in query.lower() for keyword in keywords)


# Chatbot Logic
def call_model(state: State):
    user_message = state["messages"][-1].content

    if should_use_google_search(user_message):
        links = google_search(user_message)
        scraped_content = " ".join([scrape_webpage(link) for link in links if scrape_webpage(link)])
        context = scraped_content if scraped_content else "No relevant information found."
    else:
        similar_messages = retrieve_relevant_messages(user_message, top_k=3)
        context = " ".join(similar_messages) if similar_messages else "No previous context."

    messages = [SystemMessage(content=f"Context: {context}"), HumanMessage(content=user_message)]
    response = llm.invoke(messages)

    if not should_use_google_search(user_message):
        store_message(user_message)

    return {"messages": [response]}


# Define conversation flow
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", lambda state: END)
app = workflow.compile(checkpointer=memory)


# Chat Loop
def chat():
    config = {"configurable": {"thread_id": "default"}}
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        human_message = HumanMessage(content=user_input)
        print("Chatbot: ", end="", flush=True)
        for event in app.stream({"messages": [human_message]}, config, stream_mode="updates"):
            for node, content in event.items():
                if "messages" in content:
                    for m in content["messages"]:
                        print(m.content, end=" ")
        print()


if __name__ == "__main__":
    chat()
