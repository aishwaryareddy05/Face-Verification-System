import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image, config, is_id_photo=False, image_type=None):
    """
    Main preprocessing function that handles both ID photos and selfies
    
    Args:
        image: Input image as numpy array (BGR format)
        config: Configuration dictionary
        is_id_photo: Boolean indicating if this is an ID photo
        image_type: Type of comparison being made
    
    Returns:
        Preprocessed image as numpy array
    """
    try:
        preprocessing_config = config.get('preprocessing', {})
        
        # Convert to BGR if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image
        else:
            image_bgr = image

        # Apply specific preprocessing based on image type
        if is_id_photo:
            image_bgr = enhance_id_photo(image_bgr, preprocessing_config)
        elif (image_type == "selfie_to_selfie" and 
              preprocessing_config.get("enhance_selfie", False)):
            image_bgr = enhance_selfie(image_bgr, preprocessing_config)

        # Apply general preprocessing
        image_bgr = apply_general_preprocessing(image_bgr, preprocessing_config)
        
        return image_bgr
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return image

def enhance_id_photo(image_bgr, preprocessing_config):
    """
    Enhanced ID photo preprocessing pipeline
    
    Args:
        image_bgr: Input image in BGR format
        preprocessing_config: Preprocessing configuration
    
    Returns:
        Enhanced image
    """
    try:
        # Convert to RGB for PIL operations
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 1. Intelligent upscaling for small images
        width, height = pil_image.size
        min_dimension = min(width, height)
        upscale_threshold = preprocessing_config.get('id_photo_upscale_threshold', 160)
        
        if min_dimension < upscale_threshold:
            # Calculate scale factor
            scale_factor = max(240 / min_dimension, 1.5)  # At least 1.5x upscale
            new_size = (int(width * scale_factor), int(height * scale_factor))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Upscaled ID photo from {width}x{height} to {new_size}")
        
        # Convert back to numpy for OpenCV operations
        np_image = np.array(pil_image)
        
        # 2. Histogram equalization for better contrast
        if preprocessing_config.get('enable_histogram_equalization', True):
            np_image = apply_histogram_equalization(np_image, preprocessing_config)
        
        # 3. Gamma correction for better exposure
        gamma = preprocessing_config.get('gamma_correction', 1.2)
        np_image = apply_gamma_correction(np_image, gamma)
        
        # 4. Bilateral filter for noise reduction
        bilateral_config = preprocessing_config.get('bilateral_filter', {})
        np_image = apply_bilateral_filter(np_image, bilateral_config)
        
        # 5. Unsharp masking for subtle sharpening
        if preprocessing_config.get('enable_sharpening', True):
            np_image = apply_unsharp_masking(np_image)
        
        # Convert back to BGR for InsightFace
        enhanced_bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        
        return enhanced_bgr
        
    except Exception as e:
        logger.warning(f"Could not enhance ID photo: {e}")
        return image_bgr

def enhance_selfie(image_bgr, preprocessing_config):
    """
    Mild enhancement for selfie images
    
    Args:
        image_bgr: Input image in BGR format
        preprocessing_config: Preprocessing configuration
    
    Returns:
        Enhanced selfie image
    """
    try:
        # Apply detail enhancement
        image_bgr = cv2.detailEnhance(image_bgr, sigma_s=10, sigma_r=0.15)
        
        # Apply bilateral filter for noise reduction
        bilateral_config = preprocessing_config.get('bilateral_filter', {})
        d = bilateral_config.get('d', 3)
        sigma_color = bilateral_config.get('sigma_color', 40)
        sigma_space = bilateral_config.get('sigma_space', 40)
        
        image_bgr = cv2.bilateralFilter(image_bgr, d, sigma_color, sigma_space)
        
        return image_bgr
        
    except Exception as e:
        logger.warning(f"Selfie enhancement failed: {e}")
        return image_bgr

def apply_general_preprocessing(image_bgr, preprocessing_config):
    """
    Apply general preprocessing steps
    
    Args:
        image_bgr: Input image in BGR format
        preprocessing_config: Preprocessing configuration
    
    Returns:
        Preprocessed image
    """
    try:
        # Ensure proper data type
        if image_bgr.dtype != np.uint8:
            image_bgr = np.clip(image_bgr * 255.0, 0, 255).astype(np.uint8)
        
        # Gentle denoising
        if preprocessing_config.get('enable_denoising', True):
            image_bgr = cv2.medianBlur(image_bgr, 3)
        
        # Ensure minimum resolution
        min_face_size = preprocessing_config.get('min_face_size', 112)
        image_bgr = ensure_minimum_size(image_bgr, min_face_size)
        
        return image_bgr
        
    except Exception as e:
        logger.warning(f"General preprocessing failed: {e}")
        return image_bgr

def apply_histogram_equalization(image_rgb, preprocessing_config):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image_rgb: Input image in RGB format
        preprocessing_config: Configuration dictionary
    
    Returns:
        Image with enhanced contrast
    """
    try:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        
        # Get CLAHE parameters from config
        clip_limit = preprocessing_config.get('clahe_clip_limit', 3.0)
        tile_grid_size = tuple(preprocessing_config.get('clahe_tile_grid_size', [8, 8]))
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb
        
    except Exception as e:
        logger.warning(f"Histogram equalization failed: {e}")
        return image_rgb

def apply_gamma_correction(image, gamma):
    """
    Apply gamma correction to brighten/darken image
    
    Args:
        image: Input image
        gamma: Gamma value (>1 brightens, <1 darkens)
    
    Returns:
        Gamma corrected image
    """
    try:
        # Normalize to 0-1 range
        normalized = image / 255.0
        
        # Apply gamma correction
        corrected = np.power(normalized, 1/gamma)
        
        # Convert back to 0-255 range
        result = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        logger.warning(f"Gamma correction failed: {e}")
        return image

def apply_bilateral_filter(image, bilateral_config):
    """
    Apply bilateral filter for noise reduction while preserving edges
    
    Args:
        image: Input image
        bilateral_config: Configuration for bilateral filter
    
    Returns:
        Filtered image
    """
    try:
        d = bilateral_config.get('d', 5)
        sigma_color = bilateral_config.get('sigma_color', 50)
        sigma_space = bilateral_config.get('sigma_space', 50)
        
        filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        return filtered
        
    except Exception as e:
        logger.warning(f"Bilateral filtering failed: {e}")
        return image

def apply_unsharp_masking(image, sigma=2.0, strength=1.5, threshold=0):
    """
    Apply unsharp masking for subtle sharpening
    
    Args:
        image: Input image
        sigma: Gaussian blur sigma
        strength: Sharpening strength
        threshold: Threshold for sharpening
    
    Returns:
        Sharpened image
    """
    try:
        # Create Gaussian blur
        gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Apply unsharp masking
        sharpened = cv2.addWeighted(image, 1 + strength, gaussian, -strength, threshold)
        
        return sharpened
        
    except Exception as e:
        logger.warning(f"Unsharp masking failed: {e}")
        return image

def ensure_minimum_size(image, min_size):
    """
    Ensure image meets minimum size requirements
    
    Args:
        image: Input image
        min_size: Minimum dimension size
    
    Returns:
        Resized image if necessary
    """
    try:
        height, width = image.shape[:2]
        
        if height < min_size or width < min_size:
            scale = max(min_size / height, min_size / width)
            new_height, new_width = int(height * scale), int(width * scale)
            
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            logger.info(f"Resized image to meet minimum requirements: {new_width}x{new_height}")
            
            return resized
        
        return image
        
    except Exception as e:
        logger.warning(f"Resize failed: {e}")
        return image

def detect_face_region(image, padding=20):
    """
    Detect face region and apply padding
    
    Args:
        image: Input image
        padding: Padding around detected face
    
    Returns:
        Tuple of (cropped_face, face_found)
    """
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return image, False
        
        # Take the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Apply padding
        height, width = image.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        
        # Crop face region
        face_region = image[y1:y2, x1:x2]
        
        return face_region, True
        
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return image, False

def resize_to_target(image, target_size=(160, 160)):
    """
    Resize image to target size while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: Tuple of (width, height)
    
    Returns:
        Resized image
    """
    try:
        target_width, target_height = target_size
        height, width = image.shape[:2]
        
        # Calculate scale to fit within target size
        scale = min(target_width / width, target_height / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create canvas with target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Center the resized image on canvas
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
        
    except Exception as e:
        logger.warning(f"Resize to target failed: {e}")
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)