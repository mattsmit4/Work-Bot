import streamlit as st
import os
from dotenv import load_dotenv
import re
import string

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
st.title("StarTech.com Products Chatbot")

# Fallback trigger phrases for vague follow-up questions
fallback_keywords = [
    "what color", "what colours", "what colour", "what type", "what kind", "how big", "how small", "how long",
    "how wide", "how tall", "how thick", "how heavy", "how many", "how much", "what size", "what sizes", "does it",
    "is it", "are they", "specs", "details", "specifications", "tech specs", "technical details",
    "what ports", "which ports", "what connectors", "which connectors", "what inputs", "what outputs",
    "how fast", "what speed", "what resolution", "is this compatible", "is this supported", "will this work",
    "can i use this", "can it", "is there", "do they", "what’s included", "what is included", "what do you get",
    "included accessories", "in the box", "what’s in the box", "what comes with", "do i need", "will it help", 
    "does it require", "will it fit", "will it keep", "what version", "any differences", "any difference", 
    "difference between"
]

# Phrases to detect user exit
farewell_keywords = [
    "thank you", "thanks", "appreciate it", "cheers", "bye", "goodbye", "see you", 
    "you’ve been helpful", "you have been helpful", "that’s all", "that is all"
]

# -------------------- CACHED RESOURCES --------------------

@st.cache_resource
def load_vector_store():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"])
    return PineconeVectorStore(index=index, embedding=embeddings)

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0.7)

vector_store = load_vector_store()
llm = load_llm()

# -------------------- UTILS --------------------

def extract_product_number(text):
    match = re.search(r"\b(?=[A-Z0-9-]{6,}\b)(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b", text)
    return match.group(0) if match else None

def is_vague_follow_up(prompt):
    clean = prompt.lower().translate(str.maketrans('', '', string.punctuation))
    return len(clean.strip().split()) <= 4 or any(kw in prompt.lower() for kw in fallback_keywords)

def is_farewell(prompt):
    return any(kw in prompt.lower() for kw in farewell_keywords)

def show_response(reply):
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append(AIMessage(reply))

# -------------------- HANDLERS --------------------

def handle_greeting(prompt):
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if not st.session_state.greeted_user and any(word in prompt.lower() for word in greeting_keywords):
        reply = (
            "Hi there! 👋 I'm here to help you find the right StarTech.com product.\n\n"
            "You can ask me about product specs, features, or tell me what you're trying to do and I’ll recommend something that fits."
        )
        show_response(reply)
        st.session_state.greeted_user = True
        return True
    return False

def handle_farewell(prompt):
    if is_farewell(prompt):
        show_response("Thank you for reaching out! If you have any more questions or need further assistance, feel free to ask. I'm here to help!")
        return True
    return False

def handle_explicit_product(prompt):
    product_number = extract_product_number(prompt)
    if product_number:
        docs_with_scores = vector_store.similarity_search_with_score(prompt, k=1)
        if docs_with_scores:
            top_doc, score = docs_with_scores[0]
            st.session_state.last_product_number = top_doc.metadata.get("product_number", "")
            st.session_state.last_context = top_doc.page_content
            st.session_state.last_score = score
        return True
    return False

def handle_vague_followup(prompt):
    if is_vague_follow_up(prompt):
        if not st.session_state.last_context:
            reply = (
                "Could you tell me a bit more about what you're looking for? For example:\n"
                "- Is there a specific problem you are trying to solve?\n"
                "- What specs in a product are important to you?\n"
                "- What do you want the product to connect to or be compatible with?\n\n"
                "You can also mention a product number if you already have one in mind."
            )
            show_response(reply)
            return True
        else:
            st.session_state.last_score = 1.0
    return False

def handle_descriptive_query(prompt):
    docs_with_scores = vector_store.similarity_search_with_score(prompt, k=1)
    if docs_with_scores:
        top_doc, score = docs_with_scores[0]
        if score >= 0.4:
            st.session_state.last_product_number = top_doc.metadata.get("product_number", "")
            st.session_state.last_context = top_doc.page_content
            st.session_state.last_score = score
        return True
    return False

# -------------------- SESSION STATE --------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            "You are a StarTech.com assistant. Always be friendly and professional. "
            "You only answer questions about StarTech.com products using the provided context. "
            "Do not mention or recommend products from any other company, supplier, or brand — only StarTech.com. "
            "Do not make up information. If you’re unsure, ask for clarification. "
            "Stay helpful, polite, and focused on StarTech.com solutions."
        )
    ]
if "last_product_number" not in st.session_state:
    st.session_state.last_product_number = ""
if "last_context" not in st.session_state:
    st.session_state.last_context = ""
if "last_score" not in st.session_state:
    st.session_state.last_score = None
if "greeted_user" not in st.session_state:
    st.session_state.greeted_user = False

# -------------------- CHAT FLOW --------------------

# Render previous messages
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

prompt = st.chat_input("Ask about a StarTech.com product")

if prompt:
    st.session_state.messages.append(HumanMessage(prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    if handle_greeting(prompt): st.stop()
    if handle_farewell(prompt): st.stop()
    if handle_explicit_product(prompt): pass
    elif handle_vague_followup(prompt): st.stop()
    else: handle_descriptive_query(prompt)

    if not st.session_state.last_context:
        show_response(
            "Could you tell me a bit more about what you're looking for? For example:\n"
            "- Is there a specific problem you are trying to solve?\n"
            "- What specs in a product are important to you?\n"
            "- What do you want the product to connect to or be compatible with?\n\n"
            "You can also mention a product number if you already have one in mind."
        )
        st.stop()

    # 🧠 Debug
    print("\n--- Debug Info ---")
    print(f"User Prompt: {prompt}")
    print(f"Product Number: {st.session_state.last_product_number}")
    print(f"Similarity Score: {st.session_state.last_score if st.session_state.last_score is not None else 'N/A'}")
    print("Context Retrieved:")
    print(st.session_state.last_context)
    print("-------------------\n")

    # Send context to GPT
    messages = (
        st.session_state.messages +
        [SystemMessage(f"This is the specification for product {st.session_state.last_product_number}:\n{st.session_state.last_context}")] +
        [HumanMessage(prompt)]
    )

    reply = llm.invoke(messages).content

    # Detect multiple product numbers in response
    product_numbers_in_reply = re.findall(r"\b(?=[A-Z0-9-]{6,}\b)(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b", reply)
    unique_product_numbers = list(set(product_numbers_in_reply))
    if len(unique_product_numbers) > 1:
        reply += "\n\n**Which product would you like to know more about?**"
        print(f"[⚠️ Multiple product numbers mentioned]: {unique_product_numbers}")

    show_response(reply)
