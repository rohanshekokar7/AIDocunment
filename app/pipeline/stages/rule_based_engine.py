"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from app.pipeline.interfaces.slm_engine import SLMEngine
from app.pipeline.models import DocumentContext
import re

class RuleBasedEngine(SLMEngine):
    def classify(self, document_context: DocumentContext) -> dict:
        """
        A fast, deterministic rule-based classifier using keyword matching.
        """
        from app.core.logging import logger
        
        # Aggregate all text from all pages
        full_text = ""
        for page in document_context.pages:
            for tb in page.raw_text_blocks:
                full_text += tb.text.lower() + " "
                
        logger.info(f"Rule-Based OCR Text Sample: {full_text[:200]}...")
                
        # Define rules (Weights can be added by using longer phrases)
        rules = {
            "Aadhaar Card": ["aadhaar", "uidai", "government of india", "mera aadhaar", "help@uidai.gov.in"],
            "PAN Card": ["income tax department", "permanent account number", "pan card", "father's name", "govt. of india"],
            "Passport": ["passport", "republic of india", "given name(s)", "surname", "nationality", "place of issue", "date of expiry"],
            "Driving Licence": ["driving licence", "transport department", "dl no", "vehicle class", "valid till", "rto"],
            "Voter ID": ["election commission of india", "epic", "elector's photo identity card", "voter id", "electoral registration officer"],
            "Birth Certificate": ["birth certificate", "certificate of birth", "date of birth", "place of birth", "mother's name", "father's name"],
            "Death Certificate": ["death certificate", "certificate of death", "date of death", "place of death", "cause of death"],
            "Marriage Certificate": ["marriage certificate", "certificate of marriage", "solemnized", "registrar of marriages"],
            "Income Certificate": ["income certificate", "annual income", "tahsildar", "revenue officer", "valid for the year"],
            "Caste Certificate": ["caste certificate", "scheduled caste", "scheduled tribe", "other backward class", "community certificate"],
            "Domicile Certificate": ["domicile certificate", "resident of", "residential certificate", "tehsildar"],
            "EWS Certificate": ["economically weaker section", "ews certificate", "gross annual income"],
            "Disability Certificate": ["disability certificate", "person with disability", "handicapped", "medical board", "percentage of disability"],
            "10th Marksheet": ["secondary school examination", "10th", "sslc", "matriculation", "high school", "board of secondary education", "marks obtained"],
            "12th Marksheet": ["senior secondary", "12th", "hsc", "intermediate", "higher secondary", "pre-university"],
            "Degree Certificate": ["degree certificate", "university", "bachelor degree", "master degree", "convocation", "passed the examination", "first class"],
            "Semester Marksheet": ["semester marksheet", "grade card", "cgpa", "sgpa", "statement of marks"],
            "Transfer Certificate": ["transfer certificate", "school leaving certificate", "conduct", "character"],
            "Bonafide Certificate": ["bonafide certificate", "bona fide", "is a student of this institution", "studying in class"],
            "Admit Card": ["admit card", "hall ticket", "roll number", "examination centre", "reporting time", "invigilator"],
            "Salary Slip": ["salary slip", "payslip", "earnings", "deductions", "basic pay", "hra", "net pay", "provident fund", "gross earnings", "take home"],
            "Bank Statement": ["bank statement", "account statement", "deposit", "withdrawal", "closing balance", "ifsc", "statement of account"],
            "Bank Passbook": ["passbook", "account number", "ifsc", "micr", "branch code", "customer id"],
            "Cancelled Cheque": ["cancelled", "cheque", "pay", "rupees", "bearer", "a/c payee"],
            "Form 16": ["form 16", "certificate under section 203", "tax deducted at source", "tds", "part a", "part b", "employer"],
            "Income Tax Return": ["income tax return", "itr", "assessment year", "taxable income", "computation"],
            "GST Invoice": ["gst invoice", "cgst", "sgst", "igst", "gstin", "hsn", "sac", "tax invoice", "taxable value"],
            "Medical Prescription": ["prescription", "clinic", "symptoms", "medication", "tablet", "syrup", "dosage"],
            "Lab Report": ["lab report", "pathology", "test report", "hemoglobin", "reference range", "result", "biological reference interval"],
            "Hospital Discharge Summary": ["discharge summary", "hospital", "date of admission", "date of discharge", "diagnosis", "course in hospital", "treatment given"],
            "OPD Slip": ["opd", "outpatient", "consultation", "patient id", "uhid", "vitals"],
            "Vaccination Certificate": ["vaccination certificate", "covid-19", "dose", "vaccinated", "beneficiary", "vaccine name"],
            "Appointment Letter": ["appointment letter", "terms of employment", "appointed", "probation period", "notice period", "reporting manager"],
            "Offer Letter": ["offer letter", "offered the position", "compensation package", "ctc"],
            "Experience Letter": ["experience letter", "experience certificate", "relieving", "worked with us from", "to whomsoever it may concern", "duties and responsibilities"],
            "Employment Contract": ["employment contract", "agreement of employment", "probation", "confidentiality", "non-compete"],
            "Affidavit": ["affidavit", "notary", "deponent", "solemnly affirm", "oath", "sworn", "verification"],
            "Court Order": ["court order", "in the court of", "judgement", "petition", "respondent", "plaintiff", "defendant", "hon'ble judge"],
            "FIR": ["first information report", "police station", "indian penal code", "ipc", "informant", "accused"],
            "Rent Agreement": ["rent agreement", "lease agreement", "tenant", "landlord", "monthly rent", "security deposit", "premises"],
            "Sale Deed": ["sale deed", "purchaser", "vendor", "property", "stamp duty", "consideration amount", "schedule of property", "sub-registrar"],
            "Electricity Bill": ["electricity bill", "energy bill", "power distribution", "meter number", "kwh", "due date", "bill amount"],
            "Water Bill": ["water bill", "jal board", "water supply", "consumer number", "water consumption"],
            "Gas Bill": ["gas bill", "mahanagar gas", "indane", "bharat gas", "consumer number", "cylinder", "png"],
            "Mobile Bill": ["mobile bill", "telecom", "postpaid", "airtel", "jio", "vodafone", "bsnl", "data usage", "call charges"],
            "Train Ticket": ["train ticket", "irctc", "pnr", "coach", "berth", "passenger ticket", "tatkal", "departure", "arrival"],
            "Flight Ticket": ["flight ticket", "boarding pass", "airline", "flight no", "departure", "arrival", "seat", "terminal", "gate", "baggage"],
            "Resume": ["resume", "curriculum vitae", "cv", "objective", "projects", "education", "experience", "skills", "languages known"],
            "Receipt / Invoice": ["receipt", "invoice", "amount due", "subtotal", "purchase", "payment", "bill to", "ship to", "cash memo"],
            "Leave Letter": ["leave letter", "leave application", "absence", "request for leave", "sick leave", "casual leave", "maternity leave", "respected sir/madam"],
            "Joining / HR Report": ["joining report", "hr", "employee", "joining date", "department", "reporting to", "employee id"],
            "Other (Unknown)": []
        }
        
        # Normalize whitespace in full_text to help with matching
        full_text = re.sub(r'\s+', ' ', full_text)
        
        # Count hits
        scores = {doc_type: 0 for doc_type in rules}
        
        # PASS 1: Strict word boundary matching
        for doc_type, keywords in rules.items():
            for kw in keywords:
                # Use word boundaries so that "pan" doesn't match inside "company"
                occurrences = len(re.findall(rf'\b{re.escape(kw)}\b', full_text))
                # Give more weight to longer phrases
                weight = 3 if " " in kw else 1
                scores[doc_type] += occurrences * weight
                
        # Find the max score
        best_match = max(scores, key=scores.get)
        
        # PASS 2: If strict matching failed completely, try aggressive loose matching
        if scores[best_match] == 0:
            logger.info("Pass 1 Strict Matching failed. Attempting Pass 2 Loose Matching...")
            # Strip all non-alphanumeric characters to ignore OCR spacing issues (e.g. "P A N" -> "pan")
            stripped_text = re.sub(r'[^a-z0-9]', '', full_text)
            
            for doc_type, keywords in rules.items():
                for kw in keywords:
                    stripped_kw = re.sub(r'[^a-z0-9]', '', kw)
                    # Only do loose matching for keywords longer than 5 chars to prevent huge false positives
                    if len(stripped_kw) > 5 and stripped_kw in stripped_text:
                        occurrences = stripped_text.count(stripped_kw)
                        weight = 3 if " " in kw else 1
                        scores[doc_type] += occurrences * weight
                        
            best_match = max(scores, key=scores.get)
        
        if scores[best_match] > 0:
            doc_type = best_match
            # Find which keywords matched for the summary
            # We do a loose check for the summary just to see what fired
            stripped_text = re.sub(r'[^a-z0-9]', '', full_text)
            matched_kws = [kw for kw in rules[best_match] if re.sub(r'[^a-z0-9]', '', kw) in stripped_text]
            
            summary = f"Rule-based match ({scores[best_match]} pts): Found keywords {matched_kws} associated with {best_match}."
            logger.info(f"Rule-based matched {best_match} with keywords {matched_kws}")
            
            # Calculate a proper percentage: Target score for 100% confidence is 4 points
            # 1 pt = 25%, 2 pts = 50%, 3 pts = 75%, 4+ pts = 100% (capped at 0.99)
            target_score = 4.0
            calculated_percentage = scores[best_match] / target_score
            confidence = min(calculated_percentage, 0.99)
        else:
            doc_type = "Other / Unknown"
            summary = "Rule-based approach could not confidently match any keywords."
            logger.warning("Rule-based classification failed to find any matching keywords.")
            confidence = 0.4
            
        # Very simple script detection for Hindi/Marathi
        devanagari_chars = sum(1 for char in full_text if '\u0900' <= char <= '\u097F')
        if devanagari_chars > 10:
            language = "Hindi / Marathi"
            if doc_type == "Other / Unknown":
                summary = "Rule-based approach detected Hindi/Marathi script but couldn't find English keywords."
        else:
            language = "English"
            
        return {
            "document_type": doc_type,
            "summary": summary,
            "language": language,
            "confidence": confidence
        }
