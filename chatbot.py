import streamlit as st
from streamlit_chat import message as st_chat
from pypdf import PdfReader
import os

from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.openai import OpenAI
import io
from urllib.request import Request, urlopen


os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

st.markdown(f'''<style>.sidebar .sidebar-content {{width: 375px;}}</style>''', unsafe_allow_html=True)

st.sidebar.title("Fine Print GPT")
st.sidebar.subheader("Talk to your credit cards")

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hi! I'm Fine Print GPT bot. Happy to assist your financial health."]
if "past" not in st.session_state:
    st.session_state['past'] = ["Hi"]

uploaded_file = st.sidebar.file_uploader("Please get started by uploading a credit card terms PDF file")

other_options = st.sidebar.selectbox(
    'Alternatively, you can load one of the prepared credit terms below:',
    ('', 'Affirm', 'Wheatland Bank', 'Capital One for Williams Sonoma', 'Zions Credit Card'), index=0)

st.sidebar.write("""
Some example questions:
- 1. tell me all the fees
- 2. how do they charge late fees
- 3. when will they change APRs
""")

st.sidebar.write("""
*Disclaimer*

This is a personal research project [@jerrycxu](https://twitter.com/jerrycxu). 
Credit Card Terms are taken from CFPB Database and belong to issuing banks. 
The responses from the chat bot do not constitute legal or financial advice. 
""")

CHAT_PROMPT = RichPromptTemplate = (
    """We have provided context information below. \n"
    "---------------------\n"
    "{{context_str}}"
    "\n---------------------\n"
    "Given this information, please answer the question: {{query_str}}\n"""
)

# QA_PROMPT_TMPL = (
#     "We have provided context information below. \n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "Given this information, please answer the question: {query_str}\n"
# )
# QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

# CHAT_PROMPT = RichPromptTemplate(RICH_TEMPLATE) ## A- Probably not functional and inaccurate usage

bank_to_urls = {
    'Affirm': 'https://filebin.net/n73hwbtk00fandd1/affirm_terms.pdf',
    'Zions Credit Card': 'https://files.consumerfinance.gov/a/assets/credit-card-agreements/pdf/QCCA/ZIONS_BANCORPORATION_NATIONAL_ASSOCIATION/Consumer_Credit_Card_Agreement_and_Disclosure_Statement_2022.01.03.pdf',
    'Capital One for Williams Sonoma': 'https://files.consumerfinance.gov/a/assets/credit-card-agreements/pdf/QCCA/CAPITAL_ONE_NATIONAL_ASSOCIATION/credit-card-agreement-for-williams-sonoma-key-rewards-visa-pottery-barn-key-rewards-visa-west-elm-key-rewards-visa-key-rewards-visa-in-capital-one-na.pdf',
    'Wheatland Bank': 'https://files.consumerfinance.gov/a/assets/credit-card-agreements/pdf/QCCA/WHEATLAND_BANK/WEB_ONLY_Cardholder_Agreement_Platinum_Visa.pdf',
}


def initialize_chatgpt_with_pdf(selected_file):
    reader = PdfReader(selected_file)
    docs = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        curr_text = f"We have {len(reader.pages)} pages of credit card terms. The following text are from " \
                    f"page {idx+1}: \n" \
                    f"{text}"
        docs.append(Document(text=curr_text))
    llm = OpenAI(
        model="gpt-4o",
        temperature=0.1
    )  # new model calling schema for LLMPredictor
    print(docs)
    index = VectorStoreIndex.from_documents(docs, llm_predictor=llm) # "GPTSimpleVectorIndex" is no longer supported
    return index

# def beautifulsoup_html_to_text(url):
#     url_docs = bsw.load_data(url)
#     print(url_docs[0].text)



if uploaded_file or other_options:
    if 'index' not in st.session_state:
        if uploaded_file:
            pdf_file = uploaded_file
        else:
            if other_options == 'Affirm':
                with open("./pdfs/affirm_terms.pdf", 'rb') as file:
                    pdf_file = io.BytesIO(file.read())

            else:
                url = bank_to_urls[other_options]
                st.session_state['pdf_url'] = url
                remote_file = urlopen(Request(url)).read()
                pdf_file = io.BytesIO(remote_file)
        index = initialize_chatgpt_with_pdf(pdf_file)
        st.session_state['index'] = index
    else:
        index = st.session_state['index']

    if st.session_state.get('pdf_url'):
        st.write(f"Click [here]({st.session_state.get('pdf_url')}) to download the PDF file.")
    content = st.text_input("You: ", "", key="input", disabled=False)
    if content:
        query_engine = index.as_query_engine()
        response = query_engine.query(CHAT_PROMPT)
        st.session_state.past.append(content)
        st.session_state.generated.append(response.response)
else:
    content = st.text_input("You: ", "Please submit a PDF file or choose one of the pre-loaded PDFs", key="input",
                            disabled=True)


if st.session_state.generated:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st_chat(st.session_state['generated'][i], key=str(i))
        st_chat(st.session_state['past'][i], is_user=True, key=str(i) + "_user")
