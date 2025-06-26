import cv2
import numpy as np
import insightface
from pathlib import Path
from insightface.app import FaceAnalysis
from PIL import Image, ImageEnhance, ImageFilter
import logging
from scipy.spatial.distance import cosine
import onnxruntime as ort
from preprocess import preprocess_image
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceMatcher:
    def __init__(self, config):
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        self.model = None
        self._initialize_model()
    def _initialize_model(self):
        """
        Initialize InsightFace model with optimal settings
        """
        try:
            # Set providers for ONNX runtime (CPU optimized)
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers().__contains__('CUDAExecutionProvider'):
                providers.insert(0, 'CUDAExecutionProvider')
                logger.info("CUDA available, using GPU acceleration")
        
            # Set model path explicitly to prevent runtime download
            model_root = Path("/app/models/insightface").as_posix()

            # Initialize InsightFace app with buffalo_l model 
            self.model = FaceAnalysis(
                name='buffalo_l', 
                root=model_root,  
                providers=providers,
                allowed_modules=['detection', 'recognition']
            )
        
            # Prepare the model with input size optimization
            input_size = self.config.get('model_input_size', 640)
            self.model.prepare(ctx_id=0, det_size=(input_size, input_size))
        
            logger.info(f"InsightFace model initialized successfully with input size {input_size}x{input_size}")
        
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace model: {e}")
            logger.info("Falling back to basic model configuration...")
            try:
                self.model = FaceAnalysis(
                    root="/app/models/insightface",  # 
                    providers=['CPUExecutionProvider']
                )
                self.model.prepare(ctx_id=0, det_size=(320, 320))
                logger.info("Fallback model initialized")
            except Exception as e2:
                logger.error(f"Complete model initialization failure: {e2}")
                raise RuntimeError("Cannot initialize InsightFace model")

    
    def detect_and_extract_features(self, image, is_id_photo=False, image_type=None):
        """
        Use InsightFace for detection and feature extraction
        """
        try:
            # Preprocess image
            processed_image = preprocess_image(image, is_id_photo,image_type)
            
            # Ensure model is initialized
            if self.model is None:
                logger.error("InsightFace model is not initialized.")
                return None, None, "model_not_initialized"

            # Use InsightFace to detect faces and extract features
            faces = self.model.get(processed_image)
            
            if len(faces) == 0:
                return None, None, "no_face_detected"
            
            if len(faces) > 1:
                logger.info(f"Multiple faces detected: {len(faces)}, selecting best quality")
                # Select face with highest detection confidence
                best_face = max(faces, key=lambda x: x.det_score)
                faces = [best_face]
            
            face = faces[0]
            
            # Extract the face embedding (512-dimensional vector for buffalo_l)
            embedding = face.normed_embedding
            
            # Get face bounding box
            bbox = face.bbox.astype(int)
            face_location = (bbox[1], bbox[2], bbox[3], bbox[0])  # Convert to (top, right, bottom, left)
            
            # Log detection quality metrics
            logger.info(f"Face detected - Detection score: {face.det_score:.3f}, "
                       f"Embedding norm: {np.linalg.norm(embedding):.3f}")
            
            return embedding, face_location, "success"
            
        except Exception as e:
            logger.error(f"Error in InsightFace detection/extraction: {e}")
            return None, None, "processing_error"
    
    def calculate_similarity_score(self, embedding1, embedding2, image_type=None):
        try:
            cosine_sim = np.dot(embedding1, embedding2)
            euclidean_dist = np.linalg.norm(embedding1 - embedding2)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            cosine_scaled = (cosine_sim + 1.0) / 2.0

        # Optional smoothing (soft sigmoid)
            if self.config.get("use_sigmoid_smoothing", True) and image_type == "selfie_to_selfie":
                cosine_scaled = 1 / (1 + np.exp(-10 * (cosine_scaled - 0.5)))

            if image_type == "selfie_to_selfie":
                combined_score = 0.92 * cosine_scaled + 0.08 * euclidean_sim
            else:
                combined_score = 0.85 * cosine_scaled + 0.15 * euclidean_sim

            combined_score = max(0.0, min(1.0, combined_score))

            return {
                'combined': combined_score,
                'cosine': cosine_sim,
                'cosine_scaled': cosine_scaled,
                'euclidean': euclidean_sim,
                'euclidean_distance': euclidean_dist
            }
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return {
            'combined': 0.0,
            'cosine': -1.0,
            'cosine_scaled': 0.0,
            'euclidean': 0.0,
            'euclidean_distance': 2.0
            }

    def adaptive_threshold_decision(self, scores, image_type, face_quality_metrics=None):
        """
        Smart threshold decision optimized for InsightFace similarity scores
        """
        try:
            base_thresholds = self.config["thresholds"]
            base_threshold = base_thresholds[image_type]
            
            # For InsightFace, we primarily use cosine similarity
            cosine_sim = scores['cosine']
            euclidean_dist = scores['euclidean_distance']
            
            # Adjust threshold based on image type and quality indicators
            adjusted_threshold = base_threshold
            
            if image_type == "id_to_selfie":
                # For ID photos, be more adaptive based on similarity patterns
                if euclidean_dist > 1.2:  # Large distance suggests different people
                    adjusted_threshold = base_threshold + 0.03
                elif euclidean_dist < 0.6:  # Small distance suggests same person
                    adjusted_threshold = base_threshold - 0.03
                    
                # Additional adjustment for very high or low cosine similarity
                if cosine_sim > 0.7:  # Very similar
                    adjusted_threshold = base_threshold - 0.02
                elif cosine_sim < 0.3:  # Very different
                    adjusted_threshold = base_threshold + 0.02
            
            # Make decision based on combined score
            is_match = scores['combined'] >= adjusted_threshold
            
            # Log decision details
            logger.info(f"Threshold decision - Combined: {scores['combined']:.3f}, "
                       f"Threshold: {adjusted_threshold:.3f}, Match: {is_match}")
            
            return is_match, adjusted_threshold
            
        except Exception as e:
            logger.error(f"Error in threshold decision: {e}")
            return False, self.config.get(image_type, 0.6)
    
    def determine_confidence_level(self, scores, is_match, threshold):
        """
        Determine confidence based on InsightFace-specific metrics
        """
        combined_score = scores['combined']
        cosine_sim = scores['cosine']
        euclidean_dist = scores['euclidean_distance']
        
        if is_match:
            # For positive matches, higher cosine similarity = higher confidence
            if cosine_sim > 0.8 and euclidean_dist < 0.5:
                return "high"
            elif cosine_sim > 0.6 and euclidean_dist < 0.8:
                return "medium"
            else:
                return "low"
        else:
            # For negative matches, lower cosine similarity = higher confidence
            if cosine_sim < 0.3 and euclidean_dist > 1.2:
                return "high"  # Very confident it's not a match
            elif cosine_sim < 0.5 and euclidean_dist > 0.9:
                return "medium"
            else:
                return "low"  # Close call
    
    def verify_faces(self, image1, image2, image_type):
        """
        Main verification function using InsightFace
        """
        try:
            # Determine which image might be an ID photo
            is_id_photo_1 = image_type == "id_to_selfie"
            is_id_photo_2 = False
            
            # Extract face embeddings from both images
            embedding1, location1, status1 = self.detect_and_extract_features(
                image1, is_id_photo_1
            )
            embedding2, location2, status2 = self.detect_and_extract_features(
                image2, is_id_photo_2
            )
            
            # Check for detection failures
            if status1 != "success":
                return {
                    "match_score": 0.0,
                    "match": False,
                    "status": "failed_detection",
                    "error": f"Image 1: {status1}",
                    "confidence_level": "high",
                    "threshold": self.config["thresholds"][image_type],
                    "face_locations": None
                }
            
            if status2 != "success":
                return {
                    "match_score": 0.0,
                    "match": False,
                    "status": "failed_detection",
                    "error": f"Image 2: {status2}",
                    "confidence_level": "high",
                    "threshold": self.config["thresholds"][image_type],
                    "face_locations": None
                }
            
            # Calculate similarity scores using InsightFace embeddings
            scores = self.calculate_similarity_score(embedding1, embedding2, image_type)
            
            # Make adaptive threshold decision
            is_match, used_threshold = self.adaptive_threshold_decision(
                scores, image_type
            )
            
            # Determine confidence level
            confidence_level = self.determine_confidence_level(
                scores, is_match, used_threshold
            )
            
            # Determine final status
            status = "verified" if is_match else "mismatch"
            
            # Log detailed results for debugging
            logger.info(f"InsightFace verification result: {status}, "
                       f"Combined Score: {scores['combined']:.3f}, "
                       f"Cosine Sim: {scores['cosine']:.3f}, "
                       f"Euclidean Dist: {scores['euclidean_distance']:.3f}, "
                       f"Threshold: {used_threshold:.3f}, "
                       f"Confidence: {confidence_level}")
            
            return {
                "match_score": round(scores['combined'], 3),
                "match": is_match,
                "status": status,
                "confidence_level": confidence_level,
                "threshold": used_threshold,
                "model_info": {
                    "model": "InsightFace-buffalo_l",
                    "embedding_dim": len(embedding1) if embedding1 is not None else 0
                },
                "detailed_scores": {
                    "cosine_similarity": round(scores['cosine'], 3),
                    "cosine_scaled": round(scores['cosine_scaled'], 3),
                    "euclidean_distance": round(scores['euclidean_distance'], 3),
                    "euclidean_similarity": round(scores['euclidean'], 3)
                },
                "face_locations": {
                    "image1": location1,
                    "image2": location2
                }
            }
            
        except Exception as e:
            logger.error(f"Error in InsightFace verification: {e}")
            return {
                "match_score": 0.0,
                "match": False,
                "status": "processing_error",
                "error": str(e),
                "confidence_level": "high",
                "threshold": self.config["thresholds"].get(image_type, 0.6),
                "face_locations": None
            }