"""
AI Document Classification System
Developed by Rohan Shekokar
"""

DOCUMENT_CLASSES = [
    "Aadhaar Card", "PAN Card", "Passport", "Driving Licence", "Voter ID", 
    "Birth Certificate", "Death Certificate", "Marriage Certificate", "Income Certificate",
    "Caste Certificate", "Domicile Certificate", "EWS Certificate", "Disability Certificate",
    "10th Marksheet", "12th Marksheet", "Degree Certificate", "Semester Marksheet",
    "Transfer Certificate", "Bonafide Certificate", "Admit Card", "Salary Slip",
    "Bank Statement", "Bank Passbook", "Cancelled Cheque", "Form 16", "Income Tax Return",
    "GST Invoice", "Medical Prescription", "Lab Report", "Hospital Discharge Summary",
    "OPD Slip", "Vaccination Certificate", "Appointment Letter", "Offer Letter",
    "Experience Letter", "Employment Contract", "Affidavit", "Court Order", "FIR",
    "Rent Agreement", "Sale Deed", "Electricity Bill", "Water Bill", "Gas Bill",
    "Mobile Bill", "Train Ticket", "Flight Ticket", "Resume", "Receipt / Invoice",
    "Leave Letter", "Joining / HR Report", "Other (Unknown)"
]

WRITING_TYPES = ["Printed", "Handwritten", "Mixed"]

def build_classification_prompt(document_json_str: str) -> str:
    return f"""### SYSTEM INSTRUCTION ###
Analyze the provided structured OCR text and layout of the document and extract the classification data according to the schema below.

Document Content (JSON format with pages and text blocks):
{document_json_str}

1. summary: Extract the key details of the document (e.g. Names, Amounts, Dates, Document IDs) and explain WHY you chose this document type in 1-2 sentences. Do this FIRST.
2. document_type: Identify the document from this EXACT list: {', '.join(DOCUMENT_CLASSES)}. 
   IMPORTANT CLASSIFICATION GUIDELINES (You MUST follow these rules):
   - Choose the most specific category available from the list.
   - Letters containing salary/compensation numbers are often Offer Letters or Joining Letters, NOT Invoices or Salary Slips.
   - If a document is a utility bill, be specific (Electricity Bill vs Water Bill).
   - If it doesn't clearly match any of the provided classes, choose "Other (Unknown)".
3. language: Identify the primary language of the text (e.g., "English", "Hindi", "Marathi", "Mixed"). If unable to determine, use "Unknown".
4. confidence: Estimate your confidence in this classification as a float between 0.0 and 1.0 (e.g., 0.95).

You MUST return ONLY a valid JSON object. Do not include any markdown formatting, explanations, or code blocks. Just the raw JSON string.

Here are some examples of the expected JSON output format and reasoning:

Example 1 (Standard Document):
{{
    "summary": "Found 'Permanent Account Number' and 'INCOME TAX DEPARTMENT'. Name is Rohan. Therefore this is a PAN Card.",
    "document_type": "PAN Card",
    "language": "English",
    "confidence": 0.98
}}

Example 2 (Joining Letter with Salary - NOT an Invoice):
{{
    "summary": "Found 'Dear Rohan', 'Date of Joining', 'Compensation', and company letterhead. Even though it mentions salary/money numbers, it is an employment document and clearly a Joining Letter.",
    "document_type": "Joining Letter",
    "language": "English",
    "confidence": 0.95
}}

Example 3 (Form with Handwritten Notes):
{{
    "summary": "Found printed text 'Patient Name: Rohan, Pathology Lab' alongside messy handwritten notes for 'Blood Pressure 120/80'. Since it contains both printed form text and handwritten doctor notes, it is Mixed.",
    "document_type": "Medical Report",
    "language": "English",
    "confidence": 0.92
}}
"""
