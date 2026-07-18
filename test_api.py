import json
from app.core.config import settings
from app.pipeline.stages.api_slm import APISLMEngine
from app.pipeline.models import DocumentContext

def run_test():
    engine = APISLMEngine()
    ctx = DocumentContext(file_name="dummy.jpg")
    ctx.aggregated_text = "INCOME TAX DEPARTMENT GOVT. OF INDIA Permanent Account Number"
    
    print("Sending request...")
    res = engine.classify(ctx)
    print(f"Result: {res}")

if __name__ == "__main__":
    run_test()
