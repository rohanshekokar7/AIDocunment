DOCUMENT_CLASSES = [
    "Aadhaar Card", "PAN Card", "Passport", "Bank Statement", "Passbook", 
    "Electricity Bill", "Water Bill", "Gas Bill", "Mobile Bill", "Invoice", 
    "GST Invoice", "Receipt", "Offer Letter", "Appointment Letter", 
    "Experience Letter", "Salary Slip", "Resume", "Medical Report", 
    "Government Document", "Other / Unknown"
]

WRITING_TYPES = ["Printed", "Handwritten", "Mixed"]

def build_classification_prompt(document_json_str: str) -> str:
    return f"""You are an expert Document Classification AI.
Analyze the provided structured OCR text and layout of the document and extract the following information.

Document Content (JSON format with pages and text blocks):
{document_json_str}

1. document_type: Identify the document from this exact list: {', '.join(DOCUMENT_CLASSES)}. 
   IMPORTANT RULE: You MUST choose the MOST SPECIFIC category possible. If the document is an "Aadhaar Card", "PAN Card", or "Passport", you MUST output that exact name. Do NOT output "Government Document" for these. ONLY use "Government Document" as a fallback if it is a government-issued document that is NOT an Aadhaar, PAN, or Passport. If it does not clearly match any, choose "Other / Unknown".
2. writing_type: Determine if the text is entirely "Printed", entirely "Handwritten", or "Mixed" (contains both). Use OCR text patterns, noise, or document context to infer this.
3. confidence: Estimate your confidence in this classification as a float between 0.0 and 1.0 (e.g., 0.95).

You MUST return ONLY a valid JSON object. Do not include any markdown formatting, explanations, or code blocks. Just the raw JSON string.

Example output:
{{
    "document_type": "PAN Card",
    "writing_type": "Printed",
    "confidence": 0.98
}}
"""
