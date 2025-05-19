from transformers import pipeline, Pipeline
from typing import List, Optional, Tuple

from src.config import settings

# Global variable to hold the loaded pipeline, to avoid reloading on every call
# This is a simple way to cache it. For more complex apps, consider a class or DI.
classification_pipeline: Optional[Pipeline] = None

def get_transformer_pipeline() -> Optional[Pipeline]:
    """
    Initializes and returns the zero-shot classification pipeline.
    Caches the pipeline in a global variable for efficiency.
    """
    global classification_pipeline
    if classification_pipeline is None and settings.ENABLE_TRANSFORMER_ESCALATION:
        try:
            print(f"Initializing transformer pipeline with model: {settings.TRANSFORMER_MODEL_NAME}")
            device = -1 # -1 for CPU

            classification_pipeline = pipeline(
                "zero-shot-classification", 
                model=settings.TRANSFORMER_MODEL_NAME,
                device=device # Specify CPU explicitly
            )
            print("Transformer pipeline initialized successfully.")
        except Exception as e:
            print(f"Error initializing transformer pipeline: {e}")
            # Optionally, disable further attempts if init fails hard
            # settings.ENABLE_TRANSFORMER_ESCALATION = False 
            return None
    return classification_pipeline

def classify_with_transformer(text: str, candidate_labels: List[str]) -> Optional[Tuple[str, float]]:
    """
    Classifies the given text using the zero-shot transformer pipeline against candidate labels.

    Args:
        text (str): The text to classify.
        candidate_labels (List[str]): A list of potential industry labels.

    Returns:
        Optional[Tuple[str, float]]: A tuple of (predicted_label, confidence_score) 
                                     if successful and above threshold, otherwise None.
    """
    if not settings.ENABLE_TRANSFORMER_ESCALATION:
        return None

    pipe = get_transformer_pipeline()
    if not pipe:
        print("Transformer pipeline not available for escalation.")
        return None
    
    # The 'other' category might not be a good candidate for zero-shot if it's too broad.
    # We might want to filter it out from candidate_labels for the transformer, 
    # as the transformer tries to find the *best fit* among candidates.
    # If the transformer itself is unsure, its scores will be low across all specific industries.
    specific_candidate_labels = [label for label in candidate_labels if label.lower() != 'other']
    if not specific_candidate_labels:
        # If only 'other' was a candidate, transformer can't help refine it to a specific industry
        return None 

    try:
        print(f"Transformer escalating for text (first 50 chars): '{text[:50]}...'")
        # The pipeline returns a dict like: 
        # {'sequence': text, 'labels': [sorted_labels], 'scores': [sorted_scores]}
        result = pipe(text, candidate_labels=specific_candidate_labels, multi_label=False) # multi_label=False for single best class
        
        if result and result['labels'] and result['scores']:
            predicted_label = result['labels'][0]
            confidence_score = result['scores'][0]
            print(f"Transformer prediction: {predicted_label} with confidence: {confidence_score:.4f}")
            
            if confidence_score >= settings.TRANSFORMER_CONFIDENCE_THRESHOLD:
                return predicted_label
            else:
                print(f"Transformer confidence {confidence_score:.4f} below threshold {settings.TRANSFORMER_CONFIDENCE_THRESHOLD}. Ignoring transformer prediction.")
                return "other" # Confidence too low
        else:
            print("Transformer returned no definitive result.")
            return None
    except Exception as e:
        print(f"Error during transformer classification: {e}")
        return None

if __name__ == "__main__":
    # Example Usage (requires OPENAI_API_KEY to be set if synthetic_data uses actual client)
    # And ensure ENABLE_TRANSFORMER_ESCALATION is True in .env or config for this test to run pipeline
    if settings.ENABLE_TRANSFORMER_ESCALATION:
        print("Testing transformer escalation module...")
        sample_text_finance = "The company announced its quarterly earnings report, showing strong growth in revenue and profits due to new investment strategies."
        sample_text_healthcare = "A new study published in the medical journal highlights the efficacy of a novel drug in treating patients with chronic respiratory diseases."
        sample_text_other = "This is a general announcement about the upcoming office picnic and potluck."
        
        candidate_labels_for_test = [cat for cat in settings.INITIAL_CATEGORIES if cat != 'other'] # Exclude 'other' for transformer test
        if not candidate_labels_for_test:
             candidate_labels_for_test = ["finance", "healthcare", "legal"] # fallback for test

        print(f"\nTesting with: '{sample_text_finance[:50]}...' (Candidates: {candidate_labels_for_test})")
        result_finance = classify_with_transformer(sample_text_finance, candidate_labels_for_test)
        print(f"Result for finance text: {result_finance}")

        print(f"\nTesting with: '{sample_text_healthcare[:50]}...' (Candidates: {candidate_labels_for_test})")
        result_healthcare = classify_with_transformer(sample_text_healthcare, candidate_labels_for_test)
        print(f"Result for healthcare text: {result_healthcare}")

        print(f"\nTesting with: '{sample_text_other[:50]}...' (Candidates: {candidate_labels_for_test})")
        result_other = classify_with_transformer(sample_text_other, candidate_labels_for_test)
        print(f"Result for other text: {result_other} (expected None or low confidence if not in candidates)")
    else:
        print("Transformer escalation is disabled. Skipping transformer module test.") 