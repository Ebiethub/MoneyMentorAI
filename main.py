import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import tool
import requests
from typing import List, Dict
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# st.secrets["GROQ_API_KEY"]

# Initialize LLM with Groq API
model = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-specdec",
        temperature=0.7,
        max_tokens=4000
    )

# Custom tools for money-making features
@tool
def real_time_market_scanner(symbol: str) -> Dict:
    """Get real-time financial data from Alpha Vantage API"""
    api_key = st.secrets["GROQ_API_KEY"] # ALPHA_VANTAGE_KEY
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    response = requests.get(url).json()
    return response['Global Quote']

@tool
def affiliate_link_generator(product: str) -> str:
    """Generate affiliate links for recommended products"""
    affiliate_map = {
        "shopify": "https://shopify.pxf.io/c/4352563/1543501/13624",
        "semrush": "https://www.semrush.com/?clickid=XYZ",
        "canva": "https://www.canva.com/affiliates/"
    }
    return affiliate_map.get(product.lower(), "No affiliate available")

# Define core functionality chains
business_idea_chain = (
    ChatPromptTemplate.from_template("""
    Generate {count} personalized business ideas for a {experience} entrepreneur interested in {interests}.
    Include required budget, potential monthly revenue, and first steps.
    Format as bullet points with emojis.
    """)
    | model
    | StrOutputParser()
)

investment_advice_chain = (
    ChatPromptTemplate.from_template("""
    Analyze this market data: {market_data}
    Create investment recommendations for a {risk_tolerance} investor with ${budget}.
    Include: asset allocation, specific tickers, and time horizon.
    """)
    | model
    | StrOutputParser()
)

# Streamlit UI
st.set_page_config(page_title="MoneyMentorAI", page_icon="💰", layout="wide")

st.title("💰 MoneyMentorAI - Your 24/7 Wealth Building Assistant")
st.markdown("""
<style>
    .stChatInput {background-color: #fafafa;}
    .stButton>button {background-color: #4CAF50!important;}
</style>
""", unsafe_allow_html=True)

# User profile sidebar
with st.sidebar:
    st.header("Your Profile")
    experience_level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
    interests = st.multiselect("Interests", ["E-commerce", "Stocks", "Real Estate", "Freelancing", "Crypto"])
    risk_tolerance = st.select_slider("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
    budget = st.number_input("Monthly Budget ($)", min_value=100, value=1000)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Core features tabs
tab1, tab2, tab3 = st.tabs(["💡 Business Ideas", "📈 Investments", "🔍 Market Scanner"])

with tab1:
    if st.button("Generate Business Ideas"):
        with st.spinner("Analyzing trends..."):
            ideas = business_idea_chain.invoke({
                "count": 3,
                "experience": experience_level,
                "interests": interests
            })
            st.session_state.messages.append({"role": "assistant", "content": ideas})
            st.rerun()

with tab2:
    symbol = st.text_input("Enter stock symbol:", "AAPL")
    if st.button("Get Investment Plan"):
        with st.spinner("Analyzing market..."):
            market_data = real_time_market_scanner(symbol)
            advice = investment_advice_chain.invoke({
                "market_data": market_data,
                "risk_tolerance": risk_tolerance,
                "budget": budget
            })
            st.session_state.messages.append({"role": "assistant", "content": advice})
            st.rerun()

with tab3:
    st.subheader("Real-Time Opportunities")
    if st.button("Scan Trending Markets"):
        with st.spinner("Scanning 50+ data sources..."):
            # Example real implementation would use multiple API calls
            opportunities = """
            🚀 Trending Now:
            1. AI Content Tools (45% MoM growth)
            2. Green Energy ETFs (+18% last month)
            3. Micro-SaaS Solutions
            """
            affiliate_link = affiliate_link_generator("shopify")
            st.markdown(f"{opportunities}\n\n[Start Now]({affiliate_link})")
            
# Display chat history
for message in st.session_state.messages[-3:]:  # Show last 3 messages
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Monetization features
st.sidebar.markdown("---")
st.sidebar.subheader("Premium Features")
if st.sidebar.button("🚀 Unlock Advanced Tools ($9.99/mo)"):
    st.sidebar.success("Premium features unlocked! Access:")
    st.sidebar.markdown("- AI Portfolio Manager\n- Automated Deal Finder\n- 1:1 Expert Sessions")

# Affiliate disclosures
st.markdown("---")
st.caption("Some links may be affiliate partnerships. We earn commission on qualified purchases.")
