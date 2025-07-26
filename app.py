# imports streamlit and setup
import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# Streamlit app title
st.title("StarTech.com Products Chatbot")

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# initialize embeddings model + vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Synonym map for fuzzy phrase matching
synonym_map = {
    # Core technical terms
    "mean time between failures": "mtbf",
    "what's in the box": "package contents",
    "what comes in the box": "package contents",
    "included in the box": "package contents",
    "package includes": "package contents",

    # Dimensions & weight
    "how big is it": "product dimensions",
    "how large is it": "product dimensions",
    "product size": "product dimensions",
    "how much does it weigh": "weight",
    "how heavy is it": "weight",
    "device weight": "weight",
    "box weight": "shipping weight",

    # Power
    "power usage": "power consumption",
    "how much power does it use": "power consumption",
    "power draw": "power consumption",
    "wattage": "power consumption",
    "power delivery": "power delivery",
    "does it charge": "power delivery",
    "fast charging": "fast charging support",

    # Compatibility
    "what os does it support": "os compatibility",
    "operating system compatibility": "os compatibility",
    "compatible with mac": "os compatibility",
    "windows support": "os compatibility",
    "linux compatible": "os compatibility",

    # Video
    "screen resolution": "max resolution",
    "max video resolution": "max resolution",
    "display resolution": "max resolution",
    "4k support": "4k display support",
    "ultra hd": "4k display support",
    "display ratio": "aspect ratio",

    # Connectivity
    "number of ports": "total ports",
    "total number of ports": "total ports",
    "how many ports": "total ports",
    "usb ports": "number of usb ports",
    "display connectors": "external ports",
    "input connector": "connector a",
    "output connector": "connector b",
    "host connection": "host connector",
    "host interface": "host connector",

    # Mounting & rack
    "is it wall mountable": "wall mountable",
    "can it be mounted": "mount options",
    "rack compatible": "rack mountable",
    "rack height": "rack height (u)",
    "vesa support": "vesa pattern",

    # Network & PoE
    "does it support poe": "poe support",
    "power over ethernet": "poe support",
    "network speed": "network speed",
    "ethernet speed": "network speed",

    # Standards / certifications
    "compliance": "compliance standards",
    "safety rating": "cable rating",
    "certifications": "whql certified",

    # Other
    "locking slot": "security slot support",
    "is it wireless": "wireless capability",
    "led light": "led indicators",
    "data rate": "max data rate",
    "max distance": "max transmission distance",
    "supported displays": "number of displays",
}

def replace_synonyms(text):
    lowered = text.lower()
    for phrase, replacement in synonym_map.items():
        if phrase in lowered:
            lowered = lowered.replace(phrase, replacement)
    return lowered

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("""
You are Dot, the official and knowledgeable virtual assistant for StarTech.com.

Your job is to recommend the most suitable StarTech.com product based on the structured product catalog and the user's technical requirements.

Use only the information retrieved from the embedded product data. Do not guess or make assumptions.

If the user provides enough technical detail (e.g., port types, display count, OS, usage scenario), retrieve the best-matching product from the catalog and include its product number and title in your response.

If multiple products are a match, recommend only the most relevant one and encourage the user to refine further if needed.

Keep your answers concise (1–3 sentences), and ask follow-up questions if the user is vague.

Do not store personal or sensitive info, and only assist with StarTech.com product selection.
"""))

if "retrieved_contexts" not in st.session_state:
    st.session_state.retrieved_contexts = []

if "last_product_chunk" not in st.session_state:
    st.session_state.last_product_chunk = ""

if "last_product_number" not in st.session_state:
    st.session_state.last_product_number = None

# Display chat messages from history
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# Chat input
prompt = st.chat_input("How can I help you?")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    greetings = ["yo", "hey", "hello", "hi", "greetings", "good morning", "good afternoon"]
    if prompt.lower().strip() in greetings:
        greeting_response = (
            "Hi there! I'm Dot, your StarTech.com product assistant. 😊\n\n"
            "How can I help you today?\n"
            "For the best results, please be specific about any technical questions or StarTech.com products that you have in mind."
        )
        with st.chat_message("assistant"):
            st.markdown(greeting_response)
            st.session_state.messages.append(AIMessage(greeting_response))
        st.stop()

    llm = ChatOpenAI(model="gpt-4o", temperature=1)

    fallback_keywords = [
        "what is", "what's", "how many", "how much", "which one", "what model", "what sku", "do you have it",
        "aspect ratio", "material", "what comes included", "in the box", "weight", "color", "colour", "what color", "what colour",
        "resolution", "ports", "features", "included", "does it have", "what's the",
        "what type", "what kind", "what cable", "tell me more", "specs", "specifications"
    ]
    is_follow_up = any(x in prompt.lower() for x in fallback_keywords)

    metadata_filter = {}
    if is_follow_up and st.session_state.last_product_number:
        st.session_state.retrieved_contexts.append(
            f"Previously referenced product number: {st.session_state.last_product_number}"
        )

    # Replace synonyms before retrieval
    rewritten_prompt = replace_synonyms(prompt)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.4,
            "filter": metadata_filter
        },
    )

    docs = retriever.invoke(rewritten_prompt)

    if not docs:
        no_results_message = (
            "Sorry, I couldn't find relevant product information for your question.\n\n"
            "You can try rephrasing or asking about a specific feature, product number, or port type."
        )
        with st.chat_message("assistant"):
            st.markdown(no_results_message)
            st.session_state.messages.append(AIMessage(no_results_message))
    else:
        docs_texts = [d.page_content[:3000] for d in docs[:3]]

        # ✅ Only update product memory if this is NOT a vague follow-up
        if not is_follow_up:
            st.session_state.last_product_chunk = docs_texts[0]
            first_doc = docs[0]
            product_id = first_doc.metadata.get("product_number")
            if product_id:
                st.session_state.last_product_number = product_id

        st.session_state.retrieved_contexts.extend(docs_texts)
        st.session_state.retrieved_contexts = st.session_state.retrieved_contexts[-3:]

        if is_follow_up and st.session_state.last_product_chunk:
            st.session_state.retrieved_contexts.insert(0, st.session_state.last_product_chunk)

        # ✅ NEW: add assistant message history to context window
        assistant_history = [m.content for m in st.session_state.messages if isinstance(m, AIMessage)]
        st.session_state.retrieved_contexts.extend(assistant_history[-2:])  # Keep recent 2 responses max

        system_prompt = """
You are Dot, the official and knowledgeable virtual assistant for StarTech.com.

Your job is to recommend the most suitable StarTech.com product based on the structured product catalog and the user's technical requirements.

Use only the information retrieved from the embedded product data. Do not guess or make assumptions.

If the user provides enough technical detail (e.g., port types, display count, OS, usage scenario), retrieve the best-matching product from the catalog and include its product number and title in your response.

If multiple products are a match, recommend only the most relevant one and encourage the user to refine further if needed.

Keep your answers concise (1–3 sentences), and ask follow-up questions if the user is vague.

Do not store personal or sensitive info, and only assist with StarTech.com product selection.
"""

        combined_context = "\n\n".join(st.session_state.retrieved_contexts)

        chat_with_context = [
            SystemMessage(system_prompt),
            *st.session_state.messages,
            HumanMessage(content=f"{prompt}\n\nContext:\n{combined_context}" +
                         (f"\n\nThe previous product being discussed is {st.session_state.last_product_number}."
                          if is_follow_up and st.session_state.last_product_number else ""))
        ]

        result = llm.invoke(chat_with_context).content

        with st.chat_message("assistant"):
            st.markdown(result)
            st.session_state.messages.append(AIMessage(result))
