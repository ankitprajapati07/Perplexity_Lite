from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from dotenv import load_dotenv
import os
from faiss_vectordb import store_message, retrieve_relevant_messages  # Import FAISS functions

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

# Initialize MemorySaver
memory = MemorySaver()


# Define chatbot state with an additional 'summary' field.
class State(MessagesState):
    summary: str


# Initialize LLM with streaming enabled.
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, streaming=True)


# Define the logic for calling the model.
def call_model(state: State):
    user_message = state["messages"][-1].content  # Get latest user input

    # Retrieve relevant past messages using FAISS
    similar_messages = retrieve_relevant_messages(user_message, top_k=3)

    # Prepare chatbot context
    context = " ".join(similar_messages) if similar_messages else "No previous context."
    messages = [SystemMessage(content=f"Context: {context}")] + state["messages"]

    # Call LLM
    response = llm.invoke(messages)

    # Store new message in FAISS
    store_message(user_message)

    return {"messages": [response]}


def should_continue(state: State) -> Literal[END]:
    return END  # Never summarize


# Build the conversation workflow.
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
app = workflow.compile(checkpointer=memory)


# Helper function to print the streaming updates.
def print_update(update):
    for node, content in update.items():
        if "messages" in content:
            for m in content["messages"]:
                print(m.content, end=" ")
        if "summary" in content:
            print(content["summary"], end=" ")


# Chat loop using app.stream with the proper config.
def chat():
    # Provide the required configurable key.
    config = {"configurable": {"thread_id": "default"}}
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "byy", "by"]:
            print("Chatbot: Goodbye!")
            break
        # Wrap user input as a HumanMessage.
        human_message = HumanMessage(content=user_input)
        print("Chatbot: ", end="", flush=True)
        # Use the stream interface with stream_mode "updates".
        for event in app.stream({"messages": [human_message]}, config, stream_mode="updates"):
            print_update(event)
        print()  # Newline after each response.


if __name__ == "__main__":
    chat()
