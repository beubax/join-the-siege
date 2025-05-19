import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from src.classifier.transformer_escalation import classify_with_transformer
import os
import json
from typing import List, Tuple, Optional, Dict, Any

from src.config import settings
from src.classifier.synthetic_data import generate_and_store_synthetic_data_for_industry, SYNTHETIC_DATA_DIR

class TextClassifier:
    def __init__(self, model_path: Path = settings.MODEL_PATH, vectorizer_path: Path = settings.VECTORIZER_PATH):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.vectorizer = None
        self.model = None
        
        self.industry_to_id: Dict[str, int] = {}
        self.id_to_industry: Dict[int, str] = {}
        
        self._load_vectorizer()
        self._load_model() 
        self._load_or_initialize_mapping()

    def _load_vectorizer(self):
        if self.vectorizer_path.exists():
            self.vectorizer = joblib.load(self.vectorizer_path)
            print(f"Vectorizer loaded from {self.vectorizer_path}")
        else:
            self.vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False, stop_words='english')
            print("New HashingVectorizer initialized.")
    
    def _save_vectorizer(self):
        if self.vectorizer:
            joblib.dump(self.vectorizer, self.vectorizer_path)
            print(f"Vectorizer saved to {self.vectorizer_path}")

    def _load_model(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            model_type = settings.ML_MODEL_TYPE
            print(f"No existing model found. Initializing a new '{model_type}' model.")
            if model_type == "SGDClassifier":
                self.model = SGDClassifier(loss='log_loss', random_state=42, warm_start=True)
            elif model_type == "MultinomialNB":
                self.model = MultinomialNB()
            else:
                print(f"Warning: Unknown ML_MODEL_TYPE '{model_type}'. Defaulting to SGDClassifier.")
                self.model = SGDClassifier(loss='log_loss', random_state=42, warm_start=True)
            print(f"New {model_type} model instance created.")

    def _save_model(self):
        if self.model:
            joblib.dump(self.model, self.model_path)
            print(f"Model saved to {self.model_path}")

    def is_fitted(self) -> bool:
        # For SGDClassifier, `coef_` is a good indicator. For MultinomialNB, `class_log_prior_`.
        # A more general check is if `classes_` attribute exists and is populated.
        return hasattr(self.model, 'classes_') and self.model.classes_ is not None and len(self.model.classes_) > 0


    def _load_or_initialize_mapping(self) -> None:
        """Loads industry-ID mapping from file or initializes a new one."""
        if settings.INDUSTRY_MAPPING_PATH.exists():
            try:
                with open(settings.INDUSTRY_MAPPING_PATH, 'r') as f:
                    loaded_mapping: Dict[str, int] = json.load(f)
                    self.industry_to_id = loaded_mapping
                    self.id_to_industry = {int(v): k for k, v in self.industry_to_id.items()} # Ensure IDs are int
                print(f"Industry mapping loaded from {settings.INDUSTRY_MAPPING_PATH}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading industry mapping: {e}. Initializing a new one.")
                self._initialize_new_mapping()
        else:
            print("No industry mapping file found. Initializing a new one.")
            self._initialize_new_mapping()

    def _initialize_new_mapping(self) -> None:
        """Initializes a new mapping based on INITIAL_CATEGORIES."""
        self.industry_to_id = {}
        self.id_to_industry = {}
        for idx, category_name in enumerate(settings.INITIAL_CATEGORIES):
            if idx < settings.MAX_POTENTIAL_INDUSTRIES:
                if category_name not in self.industry_to_id:
                    self.industry_to_id[category_name] = idx
                    self.id_to_industry[idx] = category_name
            else:
                print(f"Warning: Ran out of MAX_POTENTIAL_INDUSTRIES slots ({settings.MAX_POTENTIAL_INDUSTRIES}) while processing INITIAL_CATEGORIES. Category '{category_name}' and subsequent ones not added to initial map.")
                break
        self._save_mapping()

    def _save_mapping(self) -> None:
        """Saves the current industry_to_id mapping to a JSON file."""
        try:
            with open(settings.INDUSTRY_MAPPING_PATH, 'w') as f:
                json.dump(self.industry_to_id, f, indent=4)
            # print(f"Industry mapping saved to {settings.INDUSTRY_MAPPING_PATH}") 
        except IOError as e:
            print(f"Error saving industry mapping: {e}")

    def _get_or_assign_id(self, category_name: str) -> int:
        """
        Returns the ID for a category name. If the category is new,
        assigns a new ID, updates mappings, and saves the mapping.
        """
        if category_name in self.industry_to_id:
            return self.industry_to_id[category_name]
        else:
            next_id = -1
            for i in range(settings.MAX_POTENTIAL_INDUSTRIES):
                if i not in self.id_to_industry:
                    next_id = i
                    break
            
            if next_id == -1:
                raise ValueError(f"Exceeded maximum number of industries ({settings.MAX_POTENTIAL_INDUSTRIES}). Cannot add new category: {category_name}")

            self.industry_to_id[category_name] = next_id
            self.id_to_industry[next_id] = category_name
            self._save_mapping()
            print(f"Assigned new ID {next_id} to category '{category_name}'. Mapping updated.")
            return next_id

    def train(self, texts: List[str], category_names: List[str], is_initial_training: bool = False):
        """
        Trains or partially trains the model using integer IDs for categories.
        If is_initial_training is True or model is not fitted, it fits a new model.
        Otherwise, it performs incremental training (partial_fit).
        """
        if not texts or not category_names:
            print("Training data or categories are empty. Skipping training.")
            return

        if len(texts) != len(category_names):
            print("Error: texts and category_names lists must have the same length.")
            return
            
        category_ids = np.array([self._get_or_assign_id(name) for name in category_names])
        
        X_transformed = self.vectorizer.transform(texts)
        all_potential_classes = np.arange(settings.MAX_POTENTIAL_INDUSTRIES)

        if self.model is None:
            print("CRITICAL: Model object is None. Re-initializing.")
            self._load_model()
            if self.model is None:
                 print("FATAL: Model could not be initialized. Training aborted.")
                 return

        current_model_type = settings.ML_MODEL_TYPE

        if is_initial_training or not self.is_fitted():
            print(f"Performing initial model training with {len(texts)} samples using {current_model_type}.")
            self.model.partial_fit(X_transformed, category_ids, classes=all_potential_classes) 
        else:
            print(f"Performing incremental model training (partial_fit) with {len(texts)} samples using {current_model_type}.")
            self.model.partial_fit(X_transformed, category_ids, classes=all_potential_classes)
        
        self._save_model()
        self._save_vectorizer() 
        print("Model training/update complete.")

    def predict_proba_mapped(self, text: str) -> Optional[Dict[str, float]]:
        if not self.is_fitted() or not hasattr(self.model, "predict_proba"):
            return None
        
        X_transformed = self.vectorizer.transform([text])
        try:
            proba_array = self.model.predict_proba(X_transformed)[0]
            mapped_probabilities: Dict[str, float] = {}
            
            if hasattr(self.model, 'classes_') and self.model.classes_ is not None:
                model_known_ids = self.model.classes_
                for i, class_id_obj in enumerate(model_known_ids):
                    class_id = int(class_id_obj) # Ensure it's an int for dict lookup
                    category_name = self.id_to_industry.get(class_id)
                    if category_name and i < len(proba_array): # Ensure index is within bounds
                        mapped_probabilities[category_name] = proba_array[i]
                return mapped_probabilities
            else:
                print("Warning: Model has predict_proba but no .classes_ attribute. Cannot map probabilities to names.")
                return None
        except Exception as e:
            print(f"Error during predict_proba_mapped: {e}")
            return None

    def predict_proba(self, text: str) -> Optional[Dict[str, float]]:
        """ Predicts probabilities for each known category name. """
        return self.predict_proba_mapped(text)

    def predict(self, text: str) -> Optional[str]:
        if not self.is_fitted():
            print("Warning: Model not fitted. Defaulting prediction.")
            if settings.ENABLE_TRANSFORMER_ESCALATION:
                 known_categories_for_transformer = list(self.id_to_industry.values())
                 if not known_categories_for_transformer: # Fallback if mapping is empty
                     known_categories_for_transformer = settings.INITIAL_CATEGORIES
                 print(f"Escalating to transformer with categories: {known_categories_for_transformer[:5]}...") # Log first few
                 return classify_with_transformer(text, known_categories_for_transformer)
            return "other"
        
        X_transformed = self.vectorizer.transform([text])
        try:
            predicted_id_array = self.model.predict(X_transformed)
            if not predicted_id_array.size:
                print("Warning: Model predict() returned empty. Defaulting to 'other'.")
                return "other"
            predicted_id = int(predicted_id_array[0])
        except Exception as e:
            print(f"Error during model prediction: {e}. Defaulting to 'other'.")
            return "other"

        predicted_category_name = self.id_to_industry.get(predicted_id, "other")

        if settings.PREDICTION_CONFIDENCE_THRESHOLD is not None and settings.PREDICTION_CONFIDENCE_THRESHOLD > 0:
            probabilities_by_name = self.predict_proba_mapped(text) 

            if probabilities_by_name:
                max_prob_for_predicted_class = probabilities_by_name.get(predicted_category_name, 0.0)

                if max_prob_for_predicted_class < settings.PREDICTION_CONFIDENCE_THRESHOLD:
                    print(f"Confidence {max_prob_for_predicted_class:.4f} for '{predicted_category_name}' is below threshold {settings.PREDICTION_CONFIDENCE_THRESHOLD}.")
                    if settings.ENABLE_TRANSFORMER_ESCALATION:
                        print("Escalating to transformer.")
                        known_categories_for_transformer = list(self.id_to_industry.values())
                        if not known_categories_for_transformer:
                             known_categories_for_transformer = settings.INITIAL_CATEGORIES
                        return classify_with_transformer(text, known_categories_for_transformer)
                    else:
                        print("Classifying as 'other'.")
                        return "other"
                else:
                    return predicted_category_name
            else: # No probabilities available
                return predicted_category_name 
        else: # Thresholding disabled
            return predicted_category_name

    def get_model_info(self) -> Dict[str, Any]:
        model_info = {
            "model_type": settings.ML_MODEL_TYPE,
            "model_fitted": self.is_fitted(),
            "vectorizer_type": type(self.vectorizer).__name__,
            "model_path": str(self.model_path),
            "vectorizer_path": str(self.vectorizer_path),
            "industry_mapping_path": str(settings.INDUSTRY_MAPPING_PATH),
            "industry_id_to_name_map": {str(k): v for k, v in self.id_to_industry.items()},
            "industry_name_to_id_map": self.industry_to_id,
            "max_potential_industries": settings.MAX_POTENTIAL_INDUSTRIES,
            "current_mapped_categories_count": len(self.id_to_industry),
        }
        if self.is_fitted() and hasattr(self.model, 'classes_') and self.model.classes_ is not None:
            active_model_classes_ids = self.model.classes_
            model_info["model_active_category_names"] = sorted(list(set(
                self.id_to_industry.get(int(id_)) for id_ in active_model_classes_ids if int(id_) in self.id_to_industry and self.id_to_industry.get(int(id_)) is not None
            )))
        else:
            model_info["model_active_category_names"] = []
        
        return model_info

classifier_instance = TextClassifier()

def train_initial_model_from_scratch():
    print("Attempting to train an initial model from scratch...")
    
    if settings.MODEL_PATH.exists():
        try: os.remove(settings.MODEL_PATH)
        except OSError as e: print(f"Error removing model: {e}")
        print(f"Removed existing model at {settings.MODEL_PATH}")
    if settings.VECTORIZER_PATH.exists():
        try: os.remove(settings.VECTORIZER_PATH)
        except OSError as e: print(f"Error removing vectorizer: {e}")
        print(f"Removed existing vectorizer at {settings.VECTORIZER_PATH}")
    if settings.INDUSTRY_MAPPING_PATH.exists():
        try: os.remove(settings.INDUSTRY_MAPPING_PATH)
        except OSError as e: print(f"Error removing mapping: {e}")
        print(f"Removed existing industry mapping at {settings.INDUSTRY_MAPPING_PATH}")

    global classifier_instance # Make sure we're re-assigning the global instance
    classifier_instance = TextClassifier(settings.MODEL_PATH, settings.VECTORIZER_PATH) # Re-instantiate

    all_texts: List[str] = []
    all_labels_names: List[str] = [] # Store category names
    num_docs_per_industry = 10 

    print(f"Checking/Generating synthetic data for base industries: {settings.INITIAL_CATEGORIES}")
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for industry_name in settings.INITIAL_CATEGORIES:
        industry_data_path = SYNTHETIC_DATA_DIR / industry_name
        industry_data_path.mkdir(parents=True, exist_ok=True) 
        
        existing_files = list(industry_data_path.glob("doc_*.txt"))
        num_existing = len(existing_files)
        generated_files_for_this_industry: List[Tuple[Path, str]] = []

        current_files_to_use = []
        if num_existing >= num_docs_per_industry:
            print(f"Found sufficient ({num_existing}) existing synthetic documents for industry: {industry_name}.")
            current_files_to_use = existing_files[:num_docs_per_industry]
        else:
            print(f"Found {num_existing} docs for {industry_name}. Need {num_docs_per_industry}. Regenerating.")
            for old_file in existing_files:
                try: os.remove(old_file) 
                except OSError: pass 
            
            # generate_and_store_synthetic_data_for_industry returns list of (Path, category_name)
            newly_generated_tuples = generate_and_store_synthetic_data_for_industry(
                industry_name=industry_name, 
                num_documents=num_docs_per_industry
            )
            current_files_to_use = [item[0] for item in newly_generated_tuples] # extract paths
        
        for file_path in current_files_to_use:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    all_texts.append(f.read())
                    all_labels_names.append(industry_name) # Use the industry_name as label
            except Exception as e:
                print(f"Error reading synthetic file {file_path}: {e}. Skipping.")

    if not all_texts or not all_labels_names:
        print("No synthetic data. Attempting fallback dummy model using INITIAL_CATEGORIES.")
        dummy_texts = [f"Text about {cat}" for cat in settings.INITIAL_CATEGORIES for _ in range(2)]
        dummy_labels_names = [cat for cat in settings.INITIAL_CATEGORIES for _ in range(2)]
        
        if not dummy_texts:
             print("CRITICAL: No INITIAL_CATEGORIES for dummy model. Aborting training.")
             return

        classifier_instance.train(dummy_texts, dummy_labels_names, is_initial_training=True)
        print("Fallback dummy model training complete.")
        return

    print(f"Total synthetic documents collected: {len(all_texts)}")
    
    unique_label_names_in_data, counts = np.unique(all_labels_names, return_counts=True)
    min_samples_for_stratify = np.min(counts) if len(counts) > 0 else 0
    
    stratify_param = all_labels_names if min_samples_for_stratify >= 2 else None
    if stratify_param is None and len(all_labels_names) > 0:
         print(f"Warning: Not enough samples for at least one class (min_samples={min_samples_for_stratify}) for stratified split. Using non-stratified split.")

    if not all_texts:
        print("No data to train on. Aborting initial training.")
        return

    X_train, X_test, y_train_names, y_test_names = train_test_split(
        all_texts, all_labels_names, test_size=0.30, random_state=42, stratify=stratify_param
    )
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    print("Training the model on the 70% training split (is_initial_training=True)...")
    classifier_instance.train(X_train, y_train_names, is_initial_training=True)

    print("Evaluating the model on the 30% test split...")
    if classifier_instance.is_fitted():
        y_pred_names = [classifier_instance.predict(text_sample) for text_sample in X_test]
        
        print("\n--- Initial Model Validation Metrics ---")
        print(f"Accuracy: {accuracy_score(y_test_names, y_pred_names):.4f}")
        print("\nClassification Report (using names):")
        
        # Use all unique names present in test and predictions for the report
        report_target_names = sorted(list(set(y_test_names + y_pred_names)))
        
        print(classification_report(y_test_names, y_pred_names, labels=report_target_names, target_names=report_target_names, zero_division=0))
        print("--- End of Validation Metrics ---\n")
    else:
        print("Model not fitted after training attempt, cannot evaluate.")

    print(f"Initial model training and evaluation process complete.")


if __name__ == "__main__":
    print("Running initial model training script directly...")
    train_initial_model_from_scratch()
    
    print("\n--- Testing Predictions with the New Model ---")
    if classifier_instance.is_fitted(): # classifier_instance is global
        test_texts = [
            "This is a document about financial technology and investments.",
            "Patient record and medical history notes.",
            "This document discusses advanced quantum computing theories and their applications in modern physics.",
            "A summary of recent legal proceedings and case law updates.",
            "Educational materials for online learning platforms."
        ]
        for text_example in test_texts:
            prediction = classifier_instance.predict(text_example)
            probabilities = classifier_instance.predict_proba(text_example)
            print(f'\nTest prediction for: "{text_example}"')
            print(f'Predicted category: {prediction}')
            if probabilities:
                print(f'Probabilities: { {k: f"{v:.4f}" for k,v in probabilities.items()} }')
        
        print("\nCurrent model info:")
        print(json.dumps(classifier_instance.get_model_info(), indent=2))
    else:
        print("Model not fitted after training attempt in __main__, cannot run tests.") 