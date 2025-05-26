# 🧾 Invoice Extractor using Gemini + OCR

This Streamlit app intelligently extracts structured information from invoices (image/PDF), using **Google Gemini 1.5 Flash** for LLM-based parsing and **Tesseract OCR** for fallback text extraction. It supports field mapping, auto-filling, fuzzy matching, JSON export, and local search/delete of saved invoices.

![Python](https://img.shields.io/badge/Built_with-Python-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-orange?style=flat-square&logo=streamlit)
![Gemini](https://img.shields.io/badge/AI_Model-Gemini_1.5_flash-ffb400?style=flat-square&logo=google)
![Status](https://img.shields.io/badge/Status-Working-brightgreen?style=flat-square)

---

## 📸 What It Does

- 🔍 Extracts data from uploaded invoice images or PDFs
- 🧠 Uses **Gemini 1.5 Flash** to understand invoice structure
- 🧾 Falls back to **Tesseract OCR** when needed
- 🧠 Auto-fills missing fields like "Buyer Name" and "Seller Name"
- 🎯 Uses fuzzy logic to match data to expected fields
- 💾 Stores invoices as local JSON
- 🔍 Sidebar search + 🔥 delete functionality
- 📥 Exports all invoices as downloadable JSON

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Generative AI (Gemini 1.5 Flash)
- **OCR**: Tesseract via `pytesseract`
- **Image Parsing**: PIL, PyMuPDF (`fitz`)
- **Storage**: Local JSON file
- **Extras**: Fuzzy matching (`difflib`), sidebar search/delete, JSON export

---
