import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_exa import ExaSearchRetriever
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

exa_retriever = ExaSearchRetriever(k=4,type="keyword", text_contents_options=True, use_autoprompt=True, include_domains=["stackoverflow.com"])
google_api_key = os.environ["GENAI_API_KEY"]
gemini = GoogleGenerativeAI(model="gemini-1.0-pro-latest",google_api_key=google_api_key)
chat_gemini = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest",google_api_key=google_api_key,convert_system_message_to_human=True)

doc_prompt = PromptTemplate.from_template(
    """<source>
    <url>{url}</url>
    <text>{text}</text>
    </source>
    """
)

doc_chain = (RunnableLambda(lambda doc:{
    "text":doc.page_content,"url":doc.metadata["url"]}) | doc_prompt)

exa_retrieval_chain = (
    exa_retriever | doc_chain.map() | ( lambda docs : "\n".join([i.text.replace("\n\n","") for i in docs]))
)

main_resolver_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system","You are an expert coding assistant which resolves Coding errors with explanation.You use xml-formatted context to research peoples's questions. Ouput format is MARKDOWN."
        ),
        (
            "human",
            """
            Tasks:
            1) Anslyze the context and understand how to solve the error.
            2) Make some simple code example other than contexts code examples. Do not use the exact same code examples and statement from the context.
            3) At the End of your Add the urls of the sources you use to solve the error.
            Error: {Error}
            ---------------
            <context>
            {context}
            </context>"""
        )
    ]
)

resolver = (
    RunnableParallel(
        {
            "Error": RunnablePassthrough(),
            "context": exa_retrieval_chain,
        }
    )
    |main_resolver_prompt
    | chat_gemini
).with_types(input_type=str)