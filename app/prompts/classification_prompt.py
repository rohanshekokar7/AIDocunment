DOCUMENT_CLASSES = [
    "Aadhaar Card", "PAN Card", "Passport", "Bank Statement", "Passbook", 
    "Electricity Bill", "Water Bill", "Gas Bill", "Mobile Bill", "Invoice", 
    "GST Invoice", "Receipt", "Offer Letter", "Appointment Letter", 
    "Experience Letter", "Salary Slip", "Resume", "Medical Report", 
    "Government Document", "Other / Unknown"
]

WRITING_TYPES = ["Printed", "Handwritten", "Mixed"]

CLASSIFICATION_PROMPT = f"""You are an expert Document Classification AI.
Analyze the provided document image(s) carefully and extract the following information.

1. document_type: Identify the document from this exact list: {', '.join(DOCUMENT_CLASSES)}. If it does not clearly match any, choose "Other / Unknown".
2. writing_type: Determine if the text is entirely "Printed", entirely "Handwritten", or "Mixed" (contains both).
3. confidence: Estimate your confidence in this classification as a float between 0.0 and 1.0 (e.g., 0.95).

You MUST return ONLY a valid JSON object. Do not include any markdown formatting, explanations, or code blocks. Just the raw JSON string.

Example output:
{{
    "document_type": "PAN Card",
    "writing_type": "Printed",
    "confidence": 0.98
}}
"""
