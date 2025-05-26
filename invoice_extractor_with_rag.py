
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import pytesseract
import io
import base64
import pandas as pd
import fitz  # PyMuPDF
import json
import difflib

# RAG
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Function to get response from Gemini without Tesseract
def get_gemini_response_without_tesseract(input_text, img, prompt):
    response = model.generate_content([input_text, img, prompt])
    return response.text

# Function to get image from PDF
def convert_pdf_to_images(pdf_file):
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img_data = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img_data)
    return images

# Function to get image details
def get_image_details(image, filename="uploaded_file"):
    image_details = {
        "format": image.format or "PNG",
        "size": image.size,
        "mode": image.mode,
        "filename": filename
    }
    byte_data = io.BytesIO()
    image.save(byte_data, format=image.format or "PNG")
    byte_data = byte_data.getvalue()
    image_details["byte_data"] = byte_data
    return image_details, image

# OCR fallback
def extract_text_with_tesseract(image):
    return pytesseract.image_to_string(image)

# Parse key-value pairs and handle name auto-fill
def parse_key_value_text(response_text):
    parsed = {}
    for line in response_text.splitlines():
        if ':' in line or '-' in line:
            sep = ':' if ':' in line else '-'
            key, value = line.split(sep, 1)
            key = key.strip().lower()
            parsed[key] = value.strip().lstrip("*• ").replace("â‚¹", "₹")

    if "buyer name" not in parsed and "billing address" in parsed:
        billing = parsed["billing address"]
        first_line = billing.strip().split("\n")[0] if "\n" in billing else billing.split(",")[0]
        parsed["buyer name"] = first_line.strip()

    if "seller name" not in parsed and "seller address" in parsed:
        seller = parsed["seller address"]
        first_line = seller.strip().split("\n")[0] if "\n" in seller else seller.split(",")[0]
        parsed["seller name"] = first_line.strip()

    return parsed

# Merge with OCR fallback
def merge_extracted_data(gemini_dict, ocr_text):
    merged = gemini_dict.copy()
    for line in ocr_text.splitlines():
        if ':' in line or '-' in line:
            sep = ':' if ':' in line else '-'
            key, value = line.split(sep, 1)
            key = key.strip().lower()
            value = value.strip().lstrip("*• ").replace("â‚¹", "₹")
            if key not in merged or not merged[key] or merged[key] == "Not Found":
                merged[key] = value
    return merged

# Fuzzy field mapping
def map_fields_with_fuzzy_matching(merged_data, field_mapping, cutoff=0.6):
    mapped = {}
    lower_merged = {k.lower(): v for k, v in merged_data.items()}
    for display_field, match_term in field_mapping.items():
        best_match = difflib.get_close_matches(match_term, lower_merged.keys(), n=1, cutoff=cutoff)
        if best_match:
            mapped[display_field] = lower_merged[best_match[0]]
        else:
            mapped[display_field] = "Not Found"
    return mapped

# Store JSON to file
def store_json(data, filename="invoices.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    existing_data.append(data)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

# Load stored invoices
def load_json(filename="invoices.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Delete invoice by field
def delete_invoice_by_field(field, value, filename="invoices.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        new_data = [entry for entry in data if str(entry.get(field, "")).lower() != value.lower()]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
        return len(data) - len(new_data)
    return 0

# RAG search from stored invoices
def rag_search(query, all_data):
    if not all_data:
        return []

    docs = [json.dumps(entry) for entry in all_data]
    vectorizer = TfidfVectorizer().fit_transform(docs + [query])
    cosine_sim = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    similar_indices = cosine_sim.argsort()[0][-3:][::-1]
    return [all_data[idx] for idx in similar_indices]

# Streamlit UI
st.set_page_config(page_title="Invoice Extractor")
st.header("Invoice Extractor")

input_text = st.text_input("Input", key="input")
upload_file = st.file_uploader("Choose an invoice (Image or PDF)", type=["jpg", "jpeg", "png", "pdf"])

images = []
if upload_file is not None:
    if upload_file.type == "application/pdf":
        images = convert_pdf_to_images(upload_file)
    else:
        image = Image.open(upload_file)
        images = [image]

submit = st.button("SUBMIT")

input_prompt = """
You will extract all the information provided from the image, which is an invoice,
and answer all the input questions using the invoice image that has been uploaded to you.
"""

if submit and images:
    for image in images:
        image_details, image_obj = get_image_details(image)
        st.image(image_obj, caption="Uploaded Invoice", use_container_width=True)

        response = get_gemini_response_without_tesseract(input_text, image_obj, input_prompt)
        st.subheader("Extracted Text")
        st.write(response)

        st.subheader("OCR Fallback Data (if needed)")
        ocr_text = extract_text_with_tesseract(image_obj)
        st.text_area("OCR Text", ocr_text, height=200)

        parsed = parse_key_value_text(response)
        merged_data = merge_extracted_data(parsed, ocr_text)

        st.subheader("Raw Merged Data")
        st.json(merged_data)

        field_mapping = {
            "Invoice Number": "invoice number",
            "Invoice Date": "invoice date",
            "Invoice Value": "invoice value",
            "Mode of Payment": "mode of payment",
            "Order Number": "order number",
            "Order Date": "order date",
            "Payment Transaction ID": "payment transaction id",
            "Payment Date & Time": "payment date",
            "Seller Name": "seller name",
            "Seller Address": "seller address",
            "PAN No": "pan",
            "GST Registration No": "gst",
            "Buyer Name": "buyer name",
            "Buyer Address": "buyer address",
            "State/UT Code": "state",
            "Item": "item",
            "Quantity": "quantity",
            "Unit Price": "unit price",
            "Tax Rate": "tax rate",
            "Tax Amount": "tax amount",
            "Total Amount": "total amount",
            "HSN Code": "hsn",
            "Place of Supply": "place of supply",
            "Place of Delivery": "place of delivery",
            "Reverse Charge": "reverse charge"
        }

        extracted_json = map_fields_with_fuzzy_matching(merged_data, field_mapping)

        st.subheader("Extracted Summary")
        st.json(extracted_json)

        store_json(extracted_json)

# Sidebar Search/Delete
st.sidebar.header("🔍 Search Invoices")
all_data = load_json()
search_field = st.sidebar.selectbox("Search by", ["Invoice Number", "Seller Name", "Buyer Name", "GST Registration No"])
search_value = st.sidebar.text_input("Enter value to search")

if search_value:
    results = [entry for entry in all_data if search_value.lower() in str(entry.get(search_field, "")).lower()]
    if results:
        st.sidebar.success(f"Found {len(results)} match(es)")
        for idx, invoice in enumerate(results):
            with st.expander(f"Invoice {idx + 1}"):
                st.json(invoice)

        if st.sidebar.button("🗑 Delete All Matches"):
            deleted = delete_invoice_by_field(search_field, search_value)
            st.sidebar.success(f"Deleted {deleted} invoice(s)")
    else:
        st.sidebar.warning("No matches found")

# RAG Section
st.sidebar.header("🔎 Ask About Invoices (RAG)")
rag_query = st.sidebar.text_input("Ask a question")

if rag_query:
    top_matches = rag_search(rag_query, all_data)
    if top_matches:
        context = "\n".join([json.dumps(inv) for inv in top_matches])
        rag_response = model.generate_content([context, rag_query])
        st.sidebar.subheader("Gemini Response")
        st.sidebar.write(rag_response.text)
    else:
        st.sidebar.write("No similar invoices found.")

# Export Button
if all_data:
    st.sidebar.download_button("📥 Export All as JSON", json.dumps(all_data, indent=4), file_name="invoices.json", mime="application/json")
