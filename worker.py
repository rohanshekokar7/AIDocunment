"""
Standalone worker script for AI Document Classification
Runs in a separate process to avoid blocking the FastAPI web server GIL.
"""

import sys
import os
import json
import time
import traceback

def main():
    if len(sys.argv) < 5:
        print("Usage: python worker.py <job_id> <tmp_path> <filename> <all_pages>")
        sys.exit(1)
        
    job_id = sys.argv[1]
    tmp_path = sys.argv[2]
    filename = sys.argv[3]
    all_pages = sys.argv[4].lower() == 'true'
    
    job_file = f"/tmp/job_{job_id}.json"
    
    try:
        # Import pipeline dependencies here so they are loaded in this subprocess
        # This keeps the main web server extremely lightweight.
        from app.pipeline.stages.basic_validator import BasicValidator
        from app.pipeline.stages.preprocessing.advanced_preprocessor import AdvancedPreprocessor
        from app.pipeline.stages.preprocessing.orientation import OrientationFilter
        from app.pipeline.stages.preprocessing.deskew import DeskewFilter
        from app.pipeline.stages.preprocessing.resize import ResizeFilter
        from app.pipeline.stages.preprocessing.noise_removal import NoiseRemovalFilter
        from app.pipeline.stages.preprocessing.contrast import ContrastFilter
        from app.pipeline.stages.preprocessing.thresholding import ThresholdFilter
        from app.pipeline.stages.paddle_ocr import PaddleOCREngine
        from app.pipeline.stages.paddle_layout import PaddleLayoutEngine
        from app.pipeline.stages.json_aggregator import JSONAggregator
        from app.pipeline.stages.api_slm import APISLMEngine
        from app.pipeline.stages.heuristic_confidence import HeuristicConfidenceEstimator
        from app.pipeline.orchestrator import PipelineOrchestrator
        
        # Initialize pipeline
        pipeline = PipelineOrchestrator(
            validator=BasicValidator(),
            preprocessor=AdvancedPreprocessor(filters=[
                OrientationFilter(),
                DeskewFilter(),
                ResizeFilter(max_edge=1800),
                NoiseRemovalFilter(),
                ContrastFilter(),
                ThresholdFilter()
            ]),
            ocr_engine=PaddleOCREngine(),
            layout_engine=PaddleLayoutEngine(),
            aggregator=JSONAggregator(),
            slm_engine=APISLMEngine(),
            confidence_estimator=HeuristicConfidenceEstimator()
        )
        
        start_time = time.time()
        
        # Process the document
        result = pipeline.process(tmp_path, filename, all_pages)
        processing_time = round(time.time() - start_time, 2)
        
        # Save success result
        response = {
            "status": "completed",
            "result": {
                "document_type": result.document_type,
                "writing_type": result.writing_type,
                "language": result.language,
                "summary": result.summary,
                "confidence": result.confidence,
                "processing_time": processing_time
            },
            "error": None
        }
        
        with open(job_file, 'w') as f:
            json.dump(response, f)
            
    except Exception as e:
        error_msg = f"Inference error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        response = {
            "status": "error",
            "result": None,
            "error": "Internal server error during classification"
        }
        with open(job_file, 'w') as f:
            json.dump(response, f)
            
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_err:
                print(f"Failed to delete tmp file {tmp_path}: {cleanup_err}")

if __name__ == "__main__":
    main()
