import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
  # Use your preferred LLM class
# Replace with your actual Groq API key

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# 1. Setup LLM
groq_api_key=api_key,
llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct",)

# 2. Define Output Schema
response_schemas = [
    ResponseSchema(name="customer_name", description="Name of the customer"),
    ResponseSchema(name="product", description="Name of the product mentioned in the review"),
    ResponseSchema(name="price", description="Price of the product, if mentioned")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 3. Define Prompt Template
prompt = PromptTemplate(
    template="""
You are an intelligent assistant that reads customer reviews and extracts key information.

Review:
{review}

Extract the following information:
- Customer name
- Product name
- Price (if mentioned)

{format_instructions}
""",
    input_variables=["review"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

# 4. Chain for Processing
def process_review(review_text):
    formatted_prompt = prompt.format(review=review_text)
    llm_response = llm.invoke(formatted_prompt)
    parsed_output = output_parser.parse(llm_response.content)
    return parsed_output

# 5. Streamlit UI
st.title("🧠 Review Extractor - Customer, Product, Price")
review_input = st.text_area("Paste the customer review below:")

if st.button("Extract Info"):
    if review_input.strip():
        with st.spinner("Analyzing..."):
            result = process_review(review_input)
            st.success("Extraction complete!")

            st.subheader("🔍 Extracted Information")
            st.write(f"**Customer Name:** {result['customer_name']}")
            st.write(f"**Product:** {result['product']}")
            st.write(f"**Price:** {result['price']}")
    else:
        st.warning("Please enter a review first.")
