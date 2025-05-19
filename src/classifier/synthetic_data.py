# Placeholder for synthetic data generation logic

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

from src.config import settings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
except Exception as e:
    client = None

SYNTHETIC_DATA_DIR = settings.BASE_DIR / "data" / "synthetic_data"
SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

def _generate_detailed_prompt_for_industry(industry_name: str, num_documents: int = 10) -> str:
    """
    Crafts a detailed prompt to ask an LLM to generate multiple diverse documents 
    for a specific industry or for a generic 'other' category.
    """
    specific_examples = f"""For example:
- If '{industry_name}' is 'healthcare', examples could include anonymized patient summaries, medical research abstracts, hospital internal memos, public health announcements, pharmaceutical trial protocols, etc.
- If '{industry_name}' is 'finance', examples could include market analysis reports, investment proposals, internal audit summaries, customer financial advice letters, regulatory compliance briefings, etc.
- If '{industry_name}' is 'legal', examples could include case briefs, client consultation summaries, legal disclaimers, contract clauses, internal training materials on new legislation, etc.
- If '{industry_name}' is 'education', examples could include course syllabi excerpts, research paper abstracts, school policy documents, student support service descriptions, grant proposals, etc.
- If '{industry_name}' is 'technology', examples could include product specification documents, software development progress reports, patent application summaries, tech blog posts about new innovations, internal IT policy updates, etc.
"""
    if industry_name.lower() == "other":
        industry_description = "a generic 'other' category, not fitting neatly into specific common industries. These documents could be miscellaneous correspondence, general announcements, non-specific reports, or texts that are hard to categorize."
        specific_examples = "For 'other', examples could include general inquiries, miscellaneous public service announcements, internal memos about non-specific topics, community event descriptions, or personal letters that might accidentally be in a business document corpus."
    else:
        industry_description = f"the '{industry_name}' industry."

    prompt = f"""Please generate {num_documents} diverse and distinct text examples of documents typically found in {industry_description}
Each document should be substantial enough for text classification (e.g., 3-7 paragraphs long).
For each document, provide only the text content of the document itself. Do not add any preamble or explanation for each document, just the raw text.

Make sure the documents are varied in nature for {industry_description}
{specific_examples}

Format the output as a JSON list of strings, where each string is a complete document text. For example:
[{{ "document_text": "Document 1 content..." }}, {{ "document_text": "Document 2 content..." }}, ...]

Ensure the JSON is well-formed.
Generate exactly {num_documents} documents.
"""
    return prompt

def generate_and_store_synthetic_data_for_industry(industry_name: str, num_documents: int = 10) -> List[Tuple[Path, str]]:
    """
    Generates synthetic text data for a given industry using a (simulated) LLM call 
    and stores each document as a .txt file.
    The label for all documents generated under an industry will be the industry_name itself.

    Args:
        industry_name (str): The name of the industry (which will also be the label).
        num_documents (int): Number of synthetic documents to generate for this industry.

    Returns:
        List[Tuple[Path, str]]: A list of (file_path, label) tuples for the generated data.
    """
    print(f"Generating synthetic data for industry: {industry_name}")
    
    prompt = _generate_detailed_prompt_for_industry(industry_name, num_documents)
 
    try:
        if not client:
            raise Exception("OpenAI client not initialized")
        
        response = client.chat.completions.create(
            model="gpt-4o", # Or your preferred model
            response_format={ "type": "json_object" }, # For structured JSON output
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        generated_json_string = response.choices[0].message.content
        print(f"Generated JSON string: {generated_json_string}")
        # The prompt asks for a list of dicts: [{"document_text": "..."}, ...]
        generated_items = json.loads(generated_json_string)["documents"]
        # Ensure it's a list of dicts as requested in prompt
        if not isinstance(generated_items, list) or not all(isinstance(item, dict) and 'document_text' in item for item in generated_items):
           raise ValueError("LLM output did not match expected JSON structure: list of {\"document_text\": ...}")
        document_texts = [item['document_text'] for item in generated_items]

    except Exception as e:
        print(f"Error calling OpenAI or parsing response for industry '{industry_name}': {e}")
        return []

    if not document_texts or len(document_texts) != num_documents:
        print(f"Warning: Expected {num_documents} documents for '{industry_name}', but received {len(document_texts)}. Skipping storage for this batch.")
        return []

    industry_data_path = SYNTHETIC_DATA_DIR / industry_name
    industry_data_path.mkdir(parents=True, exist_ok=True)

    stored_files_and_labels = []
    for i, doc_text in enumerate(document_texts):
        file_path = industry_data_path / f"doc_{i+1}.txt"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc_text)
            stored_files_and_labels.append((file_path, industry_name)) # Label is the industry name
        except IOError as e:
            print(f"Error writing synthetic document to {file_path}: {e}")
            
    print(f"Stored {len(stored_files_and_labels)} synthetic documents for industry '{industry_name}' in {industry_data_path}")
    return stored_files_and_labels

if __name__ == "__main__":
    # Example usage: Generate data for all configured industries
    print(f"Starting synthetic data generation for initial categories: {settings.INITIAL_CATEGORIES}")
    all_generated_data = []
    for industry in settings.INITIAL_CATEGORIES:
        generated_files = generate_and_store_synthetic_data_for_industry(industry, num_documents=3) # Generate 3 docs per industry for testing
        all_generated_data.extend(generated_files)
    
    print(f"\nTotal synthetic documents generated and stored: {len(all_generated_data)}")
    if all_generated_data:
        print("Example of generated data paths and labels:")
        for p, label in all_generated_data[:2]: # Print first 2
            print(f"  Path: {p}, Label: {label}")

# Example of how this might be used with the classifier:
# def add_new_industry_and_retrain(industry_name: str, industry_categories: List[str]):
#     from .ml_model import classifier_instance # Import here to avoid circular deps if any
#     
#     # 1. Update settings or a dynamic category list if necessary
#     # For example, add new categories to settings.INITIAL_CATEGORIES or a similar dynamic list
#     # This part needs careful management of how categories are tracked globally.
# 
#     # 2. Generate synthetic data
#     new_data = generate_synthetic_data_for_industry(industry_name, industry_categories)
#     if new_data:
#         texts, labels = zip(*new_data)
#         # 3. Retrain the model incrementally
#         classifier_instance.train(list(texts), list(labels), is_initial_training=False)
#         print(f"Model retrained with synthetic data for new industry: {industry_name}")
#     else:
#         print(f"No synthetic data generated for industry: {industry_name}") 