"""
AI Document Classification System
Developed by Rohan Shekokar
"""

DOCUMENT_CLASSES = [
    "Aadhaar Card", "PAN Card", "Passport", "Bank Statement", "Passbook", 
    "Electricity Bill", "Water Bill", "Gas Bill", "Mobile Bill", "Invoice", 
    "GST Invoice", "Receipt", "Offer Letter", "Appointment Letter", "Joining Letter",
    "Experience Letter", "Salary Slip", "Resume", "Medical Report", 
    "Government Document", "Other / Unknown"
]

WRITING_TYPES = ["Printed", "Handwritten", "Mixed"]

def build_classification_prompt(document_json_str: str) -> str:
    return f"""### SYSTEM INSTRUCTION ###
Analyze the provided structured OCR text and layout of the document and extract the classification data according to the schema below.

Document Content (JSON format with pages and text blocks):
{document_json_str}

1. summary: Extract the key details of the document (e.g. Company name, Employee name, Invoice Amount, Dates) and explain WHY you chose this document type in 1-2 sentences. Do this FIRST.
2. document_type: Identify the document from this exact list: {', '.join(DOCUMENT_CLASSES)}. 
   IMPORTANT CLASSIFICATION GUIDELINES (You MUST follow these rules):
   - Aadhaar Card: Look for "Aadhaar", "Government of India", 12-digit number (XXXX XXXX XXXX), "DOB", "Male/Female".
   - PAN Card: Look for "INCOME TAX DEPARTMENT", "GOVT. OF INDIA", "Permanent Account Number", 10-character alphanumeric PAN.
   - Passport: Look for "Republic of India", "Passport No.", "Nationality", "Given Name", MRZ codes (<<<).
   - Bank Statement: Look for lists of transactions, "Withdrawal", "Deposit", "Balance", "Account Number", bank logos.
   - Passbook: Look for a small booklet format, "Account No", "CIF No", "IFSC", branch details, stamped entries.
   - Electricity / Water / Gas / Mobile Bill: Look for "Consumer No", "Meter No", "Due Date", "Bill Amount", "Units Consumed", utility provider names. Classify specifically based on the utility type.
   - GST Invoice: Look for "Tax Invoice", "GSTIN", "CGST", "SGST", "HSN/SAC", "Total Amount".
   - Invoice / Receipt: Look for "Invoice No", "Date", items, quantity, total price. (Use if it lacks GST details).
   - Offer / Appointment / Experience Letter: Look for company letterhead, "Dear [Name]", "Compensation", "Role", "Date of Joining", "Relieving". IMPORTANT: Letters containing salary/compensation numbers are NOT Invoices. Choose the most accurate letter type based on context.
   - Salary Slip: Look for "Earnings", "Deductions", "Basic Pay", "HRA", "PF", "Net Pay", "Employee ID".
   - Resume: Look for "Experience", "Education", "Skills", contact details, "Objective".
   - Medical Report: Look for "Patient Name", "Age", "Test Name", "Result", "Reference Range", "Clinic/Hospital/Pathology".
   - Government Document: ONLY use this as a fallback for official government docs that are NOT Aadhaar, PAN, or Passport.
   - If it doesn't clearly match any, choose "Other / Unknown".
3. writing_type: Determine if the text is entirely "Printed", entirely "Handwritten", or "Mixed" (contains both). Use OCR text patterns, noise, or document context to infer this.
4. language: Identify the primary language of the text (e.g., "English", "Hindi", "Spanish", "Mixed"). If unable to determine, use "Unknown".
5. confidence: Estimate your confidence in this classification as a float between 0.0 and 1.0 (e.g., 0.95).

You MUST return ONLY a valid JSON object. Do not include any markdown formatting, explanations, or code blocks. Just the raw JSON string.

Here are some examples of the expected JSON output format and reasoning:

Example 1 (Standard Document):
{{
    "summary": "Found 'Permanent Account Number' and 'INCOME TAX DEPARTMENT'. Name is Rohan. Therefore this is a PAN Card.",
    "document_type": "PAN Card",
    "writing_type": "Printed",
    "language": "English",
    "confidence": 0.98
}}

Example 2 (Joining Letter with Salary - NOT an Invoice):
{{
    "summary": "Found 'Dear Rohan', 'Date of Joining', 'Compensation', and company letterhead. Even though it mentions salary/money numbers, it is an employment document and clearly a Joining Letter.",
    "document_type": "Joining Letter",
    "writing_type": "Printed",
    "language": "English",
    "confidence": 0.95
}}

Example 3 (Form with Handwritten Notes):
{{
    "summary": "Found printed text 'Patient Name: Rohan, Pathology Lab' alongside messy handwritten notes for 'Blood Pressure 120/80'. Since it contains both printed form text and handwritten doctor notes, it is Mixed.",
    "document_type": "Medical Report",
    "writing_type": "Mixed",
    "language": "English",
    "confidence": 0.92
}}
"""
