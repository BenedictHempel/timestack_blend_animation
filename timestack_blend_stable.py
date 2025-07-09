#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import subprocess
import argparse
from pathlib import Path
import logging

# Rolling window configuration
MAX_FRAMES_VISIBLE = 30
DEFAULT_BATCH_SIZE = 50
MEMORY_CACHE_DIR = "aligned_cache"
SIMPLIFIED_ALIGNMENT_MODE = False  # Set to True for translation-only alignment

def auto_align_images_batch(images, use_sift_fallback=True, max_features=5000, good_match_percent=0.15, 
                           scale_factor=0.5, simplified_mode=False):
    """
    Align a batch of images with optimizations for large sequences.
    
    Uses ORB (Oriented FAST and Rotated BRIEF) feature detector as the primary method,
    with SIFT as fallback if available and ORB fails.
    
    Args:
        images: List of images (numpy arrays) to align
        use_sift_fallback: Whether to use SIFT if ORB fails (default: True)
        max_features: Maximum number of features to detect (default: 5000)
        good_match_percent: Percentage of matches to keep (default: 0.15)
        scale_factor: Scale factor for feature detection speedup (default: 0.5)
        simplified_mode: Use translation-only alignment for speed (default: False)
    
    Returns:
        tuple: (aligned_images, transformation_matrices)
            - aligned_images: List of aligned images
            - transformation_matrices: List of homography/translation matrices used for alignment
    """
    if not images or len(images) < 2:
        logger.warning("Need at least 2 images for alignment")
        return images, []
    
    reference_image = images[0]
    aligned_images = [reference_image.copy()]  # Reference image stays unchanged
    transformation_matrices = [np.eye(3)]  # Identity matrix for reference
    
    # Scale down reference for faster feature detection if scale_factor < 1.0
    if scale_factor != 1.0 and scale_factor > 0:
        ref_height, ref_width = reference_image.shape[:2]
        scaled_width = int(ref_width * scale_factor)
        scaled_height = int(ref_height * scale_factor)
        ref_scaled = cv2.resize(reference_image, (scaled_width, scaled_height))
    else:
        ref_scaled = reference_image
        scale_factor = 1.0
    
    # Convert reference to grayscale (use scaled version for feature detection)
    if len(ref_scaled.shape) == 3:
        ref_gray = cv2.cvtColor(ref_scaled, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_scaled
    
    # Try to create ORB detector
    try:
        orb = cv2.ORB_create(nfeatures=max_features)
        detector_name = "ORB"
        logger.info(f"Using {detector_name} feature detector")
    except Exception as e:
        logger.error(f"Failed to create ORB detector: {e}")
        if use_sift_fallback:
            try:
                orb = cv2.SIFT_create(nfeatures=max_features)
                detector_name = "SIFT"
                logger.info(f"Falling back to {detector_name} feature detector")
            except Exception as e:
                logger.error(f"Failed to create SIFT detector: {e}")
                logger.error("No feature detector available, returning original images")
                return images, [np.eye(3)] * len(images)
        else:
            logger.error("ORB failed and SIFT fallback disabled, returning original images")
            return images, [np.eye(3)] * len(images)
    
    # Detect keypoints and descriptors for reference image
    try:
        ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)
        if ref_descriptors is None or len(ref_keypoints) < 10:
            logger.error("Insufficient keypoints in reference image")
            return images, [np.eye(3)] * len(images)
        
        logger.info(f"Reference image: {len(ref_keypoints)} keypoints detected")
    except Exception as e:
        logger.error(f"Failed to detect features in reference image: {e}")
        return images, [np.eye(3)] * len(images)
    
    # Create matcher
    if detector_name == "SIFT":
        # Use FLANN matcher for SIFT (better for SIFT descriptors)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        use_flann = True
    else:
        # Use BFMatcher for ORB (better for binary descriptors)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        use_flann = False
    
    # Process each image after the reference
    for i, current_image in enumerate(images[1:], 1):
        logger.info(f"Aligning image {i+1}/{len(images)}")
        
        try:
            # Scale down current image for feature detection if needed
            if scale_factor != 1.0:
                cur_height, cur_width = current_image.shape[:2]
                scaled_width = int(cur_width * scale_factor)
                scaled_height = int(cur_height * scale_factor)
                current_scaled = cv2.resize(current_image, (scaled_width, scaled_height))
            else:
                current_scaled = current_image
            
            # Convert current image to grayscale (use scaled version)
            if len(current_scaled.shape) == 3:
                current_gray = cv2.cvtColor(current_scaled, cv2.COLOR_BGR2GRAY)
            else:
                current_gray = current_scaled
            
            # Detect keypoints and descriptors for current image
            current_keypoints, current_descriptors = orb.detectAndCompute(current_gray, None)
            
            if current_descriptors is None or len(current_keypoints) < 10:
                logger.warning(f"Image {i+1}: Insufficient keypoints detected, skipping alignment")
                aligned_images.append(current_image.copy())
                transformation_matrices.append(np.eye(3))
                continue
            
            logger.info(f"Image {i+1}: {len(current_keypoints)} keypoints detected")
            
            # Match features
            if use_flann:
                # FLANN matcher requires float32 descriptors and different matching approach
                if ref_descriptors.dtype != np.float32:
                    ref_desc_float = ref_descriptors.astype(np.float32)
                    current_desc_float = current_descriptors.astype(np.float32)
                else:
                    ref_desc_float = ref_descriptors
                    current_desc_float = current_descriptors
                
                matches = matcher.knnMatch(current_desc_float, ref_desc_float, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
            else:
                # BFMatcher for ORB
                matches = matcher.match(current_descriptors, ref_descriptors)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Keep only the best matches
                num_good_matches = int(len(matches) * good_match_percent)
                good_matches = matches[:max(num_good_matches, 10)]
            
            logger.info(f"Image {i+1}: {len(good_matches)} good matches found")
            
            if len(good_matches) < 10:
                logger.warning(f"Image {i+1}: Insufficient good matches for homography, skipping alignment")
                aligned_images.append(current_image.copy())
                transformation_matrices.append(np.eye(3))
                continue
            
            # Extract matching points
            src_pts = np.float32([current_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([ref_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Scale points back to full resolution if we used scaled images
            if scale_factor != 1.0:
                src_pts = src_pts / scale_factor
                dst_pts = dst_pts / scale_factor
            
            # Compute transformation
            if simplified_mode:
                # Translation-only alignment for speed
                translation = np.mean(dst_pts - src_pts, axis=0)[0]
                homography = np.array([
                    [1, 0, translation[0]],
                    [0, 1, translation[1]],
                    [0, 0, 1]
                ], dtype=np.float32)
                mask = np.ones((len(good_matches), 1), dtype=np.uint8)
                logger.info(f"Image {i+1}: Translation-only alignment: dx={translation[0]:.2f}, dy={translation[1]:.2f}")
            else:
                # Full homography using RANSAC
                homography, mask = cv2.findHomography(
                    src_pts, dst_pts, 
                    method=cv2.RANSAC, 
                    ransacReprojThreshold=5.0,
                    confidence=0.99,
                    maxIters=2000
                )
            
            if homography is None:
                logger.warning(f"Image {i+1}: Failed to compute homography, skipping alignment")
                aligned_images.append(current_image.copy())
                transformation_matrices.append(np.eye(3))
                continue
            
            # Count inliers
            inliers = np.sum(mask)
            logger.info(f"Image {i+1}: {inliers}/{len(good_matches)} inliers in homography")
            
            if inliers < 10:
                logger.warning(f"Image {i+1}: Too few inliers, skipping alignment")
                aligned_images.append(current_image.copy())
                transformation_matrices.append(np.eye(3))
                continue
            
            # Apply perspective transformation
            height, width = reference_image.shape[:2]
            aligned_image = cv2.warpPerspective(
                current_image, homography, (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            aligned_images.append(aligned_image)
            transformation_matrices.append(homography)
            
            logger.info(f"Image {i+1}: Successfully aligned")
            
        except Exception as e:
            logger.error(f"Image {i+1}: Error during alignment: {e}")
            aligned_images.append(current_image.copy())
            transformation_matrices.append(np.eye(3))
            continue
    
    logger.info(f"Alignment complete. Successfully aligned {len([m for m in transformation_matrices[1:] if not np.allclose(m, np.eye(3))])}/{len(images)-1} images")
    
    return aligned_images, transformation_matrices


def auto_crop_images(aligned_images, transformation_matrices, min_overlap_ratio=0.1):
    """
    Automatically crop aligned images to their largest common rectangular area.
    
    This function calculates the valid region after alignment (non-black areas),
    finds the largest common rectangular area across all aligned images,
    and crops all images to this common area to ensure consistent dimensions.
    
    Args:
        aligned_images: List of aligned images (numpy arrays)
        transformation_matrices: List of transformation matrices used for alignment
        min_overlap_ratio: Minimum ratio of overlap area to original area (default: 0.1)
                          If overlap is smaller, returns original images
    
    Returns:
        tuple: (cropped_images, crop_bounds)
            - cropped_images: List of cropped images with consistent dimensions
            - crop_bounds: Dictionary with 'x', 'y', 'width', 'height' of crop region
    """
    if not aligned_images:
        logger.warning("No images provided for cropping")
        return [], {}
    
    if len(aligned_images) == 1:
        logger.info("Only one image provided, returning as-is")
        return aligned_images, {'x': 0, 'y': 0, 'width': aligned_images[0].shape[1], 'height': aligned_images[0].shape[0]}
    
    reference_image = aligned_images[0]
    height, width = reference_image.shape[:2]
    original_area = height * width
    
    logger.info(f"Finding common crop area for {len(aligned_images)} aligned images ({width}x{height})")
    
    # Calculate valid regions for each image
    valid_regions = []
    
    for i, (image, transform_matrix) in enumerate(zip(aligned_images, transformation_matrices)):
        if np.allclose(transform_matrix, np.eye(3)):
            # Identity transformation - entire image is valid
            valid_region = np.ones((height, width), dtype=np.uint8) * 255
            logger.info(f"Image {i+1}: Identity transform, entire area valid")
        else:
            # Create mask of non-black pixels (areas that have been transformed)
            if len(image.shape) == 3:
                # Color image - check if any channel is non-zero
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Create binary mask where pixel values > threshold (non-black)
            # Use a small threshold to account for interpolation artifacts
            threshold = 5
            valid_region = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            valid_region = cv2.morphologyEx(valid_region, cv2.MORPH_CLOSE, kernel)
            valid_region = cv2.morphologyEx(valid_region, cv2.MORPH_OPEN, kernel)
            
            valid_pixels = np.sum(valid_region > 0)
            logger.info(f"Image {i+1}: {valid_pixels}/{height*width} valid pixels ({valid_pixels/(height*width)*100:.1f}%)")
        
        valid_regions.append(valid_region)
    # Find intersection of all valid regions
    common_valid_region = valid_regions[0].copy()
    for valid_region in valid_regions[1:]:
        common_valid_region = cv2.bitwise_and(common_valid_region, valid_region)
    
    # Find the largest rectangular area within the common valid region
    # First, find contours of the valid region
    contours, _ = cv2.findContours(common_valid_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning("No common valid region found, returning original images")
        crop_bounds = {'x': 0, 'y': 0, 'width': width, 'height': height}
        return aligned_images, crop_bounds
    
    # Find the bounding rectangle of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Refine the rectangle to find the largest inscribed rectangle
    # This uses a more sophisticated approach to find the maximum rectangle
    crop_x, crop_y, crop_w, crop_h = find_largest_inscribed_rectangle(common_valid_region, x, y, w, h)
    
    # Check if the cropped area is reasonable
    crop_area = crop_w * crop_h
    overlap_ratio = crop_area / original_area
    
    logger.info(f"Common crop area: ({crop_x}, {crop_y}) {crop_w}x{crop_h} (area: {crop_area}, {overlap_ratio*100:.1f}% of original)")
    
    if overlap_ratio < min_overlap_ratio:
        logger.warning(f"Overlap ratio {overlap_ratio:.3f} is below minimum {min_overlap_ratio}, returning original images")
        crop_bounds = {'x': 0, 'y': 0, 'width': width, 'height': height}
        return aligned_images, crop_bounds
    
    # Crop all images to the common area
    cropped_images = []
    for i, image in enumerate(aligned_images):
        try:
            # Ensure crop coordinates are within image bounds
            crop_x_safe = max(0, min(crop_x, width - 1))
            crop_y_safe = max(0, min(crop_y, height - 1))
            crop_x2_safe = max(crop_x_safe + 1, min(crop_x + crop_w, width))
            crop_y2_safe = max(crop_y_safe + 1, min(crop_y + crop_h, height))
            
            cropped_image = image[crop_y_safe:crop_y2_safe, crop_x_safe:crop_x2_safe]
            
            if cropped_image.size == 0:
                logger.warning(f"Image {i+1}: Crop resulted in empty image, using original")
                cropped_images.append(image.copy())
            else:
                cropped_images.append(cropped_image)
                logger.info(f"Image {i+1}: Cropped to {cropped_image.shape[1]}x{cropped_image.shape[0]}")
        
        except Exception as e:
            logger.error(f"Image {i+1}: Error during cropping: {e}, using original")
            cropped_images.append(image.copy())
    
    crop_bounds = {
        'x': crop_x_safe,
        'y': crop_y_safe, 
        'width': crop_x2_safe - crop_x_safe,
        'height': crop_y2_safe - crop_y_safe
    }
    
    logger.info(f"Auto-crop complete: {len(cropped_images)} images cropped to {crop_bounds['width']}x{crop_bounds['height']}")
    
    return cropped_images, crop_bounds


def find_largest_inscribed_rectangle(binary_mask, start_x, start_y, start_w, start_h):
    """
    Find the largest inscribed rectangle within a binary mask using dynamic programming.
    
    This function uses the largest rectangle in histogram algorithm applied to
    each row of the binary mask to find the maximum rectangular area.
    
    Args:
        binary_mask: Binary mask (numpy array) where 255 = valid, 0 = invalid
        start_x, start_y, start_w, start_h: Initial bounding rectangle to search within
    
    Returns:
        tuple: (x, y, width, height) of the largest inscribed rectangle
    """
    # Crop the mask to the search region
    mask_region = binary_mask[start_y:start_y+start_h, start_x:start_x+start_w]
    
    if mask_region.size == 0:
        return start_x, start_y, 1, 1
    
    # Convert to binary (0 or 1)
    binary_region = (mask_region > 0).astype(np.int32)
    
    rows, cols = binary_region.shape
    
    # Calculate histogram for first row
    histogram = binary_region[0, :].copy()
    max_area = 0
    best_rect = (0, 0, 1, 1)
    
    for row in range(rows):
        if row > 0:
            # Update histogram: if current cell is 1, add 1 to histogram,
            # otherwise reset to 0
            for col in range(cols):
                if binary_region[row, col] == 1:
                    histogram[col] += 1
                else:
                    histogram[col] = 0
        
        # Find largest rectangle in current histogram
        area, rect = largest_rectangle_in_histogram(histogram)
        
        if area > max_area:
            max_area = area
            # Convert back to original coordinates
            rect_x, rect_w, rect_h = rect
            rect_y = row - rect_h + 1
            best_rect = (start_x + rect_x, start_y + rect_y, rect_w, rect_h)
    
    return best_rect


def create_cache_directory(cache_dir):
    """
    Create cache directory for aligned images if memory is constrained.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir

def save_aligned_image_to_cache(image, cache_dir, frame_index):
    """
    Save aligned image to disk cache.
    """
    cache_path = os.path.join(cache_dir, f"aligned_{frame_index:06d}.jpg")
    cv2.imwrite(cache_path, image)
    return cache_path

def load_aligned_image_from_cache(cache_dir, frame_index):
    """
    Load aligned image from disk cache.
    """
    cache_path = os.path.join(cache_dir, f"aligned_{frame_index:06d}.jpg")
    if os.path.exists(cache_path):
        return cv2.imread(cache_path)
    return None

def cleanup_cache_file(cache_dir, frame_index):
    """
    Remove cache file for old frame.
    """
    cache_path = os.path.join(cache_dir, f"aligned_{frame_index:06d}.jpg")
    if os.path.exists(cache_path):
        os.remove(cache_path)

def largest_rectangle_in_histogram(histogram):
    """
    Find the largest rectangular area in a histogram using stack-based algorithm.
    
    Args:
        histogram: Array of histogram heights
    
    Returns:
        tuple: (max_area, (x, width, height)) where (x, width, height) describes the rectangle
    """
    stack = []
    max_area = 0
    best_rect = (0, 1, 1)
    
    for i, h in enumerate(histogram):
        while stack and histogram[stack[-1]] > h:
            height = histogram[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            x = 0 if not stack else stack[-1] + 1
            area = height * width
            
            if area > max_area:
                max_area = area
                best_rect = (x, width, height)
        
        stack.append(i)
    
    # Process remaining bars in stack
    while stack:
        height = histogram[stack.pop()]
        width = len(histogram) if not stack else len(histogram) - stack[-1] - 1
        x = 0 if not stack else stack[-1] + 1
        area = height * width
        
        if area > max_area:
            max_area = area
            best_rect = (x, width, height)
    
    return max_area, best_rect


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SlidingWindowBuffer:
    """
    Manages a sliding window of images with optional disk caching.
    """
    def __init__(self, max_size=30, use_cache=False, cache_dir=None):
        self.max_size = max_size
        self.use_cache = use_cache
        self.cache_dir = cache_dir if cache_dir else MEMORY_CACHE_DIR
        self.images = []  # List of images in memory
        self.frame_indices = []  # Corresponding frame indices
        self.oldest_index = 0  # Index of oldest frame for cache management
        
        if self.use_cache:
            create_cache_directory(self.cache_dir)
    
    def add_image(self, image, frame_index):
        """
        Add new image to the buffer, removing oldest if at capacity.
        """
        if len(self.images) >= self.max_size:
            # Remove oldest image
            if self.use_cache:
                # Clean up cache file for removed frame
                cleanup_cache_file(self.cache_dir, self.frame_indices[0])
            
            self.images.pop(0)
            self.frame_indices.pop(0)
        
        # Add new image
        if self.use_cache:
            # Save to cache and store reference
            save_aligned_image_to_cache(image, self.cache_dir, frame_index)
            self.images.append(None)  # Placeholder, will load when needed
        else:
            self.images.append(image.copy())
        
        self.frame_indices.append(frame_index)
    
    def get_all_images(self):
        """
        Get all images in the buffer (loading from cache if needed).
        """
        if self.use_cache:
            loaded_images = []
            for i, frame_idx in enumerate(self.frame_indices):
                if self.images[i] is None:
                    # Load from cache
                    img = load_aligned_image_from_cache(self.cache_dir, frame_idx)
                    if img is not None:
                        loaded_images.append(img)
                    else:
                        logger.warning(f"Failed to load cached image for frame {frame_idx}")
                else:
                    loaded_images.append(self.images[i])
            return loaded_images
        else:
            return [img for img in self.images if img is not None]
    
    def size(self):
        return len(self.images)
    
    def cleanup(self):
        """
        Clean up cache directory.
        """
        if self.use_cache:
            for frame_idx in self.frame_indices:
                cleanup_cache_file(self.cache_dir, frame_idx)

def apply_blend_mode(accumulator, current_img, blend_mode):
    """
    Apply different blending modes between accumulator and current image
    
    Args:
        accumulator: Accumulated image as float32
        current_img: Current image as float32
        blend_mode: Blending mode string
    
    Returns:
        Blended result as float32
    """
    if blend_mode == "lighten":
        # Lighten: max(accumulator, current)
        return np.maximum(accumulator, current_img)
    
    elif blend_mode == "darken":
        # Darken: min(accumulator, current)
        return np.minimum(accumulator, current_img)
    
    elif blend_mode == "lighter_color":
        # Lighter Color: choose pixel with higher luminance
        # Convert to grayscale for luminance comparison
        acc_gray = cv2.cvtColor(accumulator.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        cur_gray = cv2.cvtColor(current_img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Create mask where current is lighter
        mask = cur_gray > acc_gray
        mask = np.stack([mask, mask, mask], axis=-1)
        
        return np.where(mask, current_img, accumulator)
    
    elif blend_mode == "darker_color":
        # Darker Color: choose pixel with lower luminance
        acc_gray = cv2.cvtColor(accumulator.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        cur_gray = cv2.cvtColor(current_img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Create mask where current is darker
        mask = cur_gray < acc_gray
        mask = np.stack([mask, mask, mask], axis=-1)
        
        return np.where(mask, current_img, accumulator)
    
    elif blend_mode == "difference":
        # Difference: abs(accumulator - current)
        return np.abs(accumulator - current_img)
    
    elif blend_mode == "multiply":
        # Multiply: (accumulator * current) / 255
        return (accumulator * current_img) / 255.0
    
    elif blend_mode == "screen":
        # Screen: 255 - ((255 - accumulator) * (255 - current)) / 255
        return 255.0 - ((255.0 - accumulator) * (255.0 - current_img)) / 255.0
    
    elif blend_mode == "overlay":
        # Overlay: combination of multiply and screen
        mask = accumulator < 128
        result = np.zeros_like(accumulator)
        
        # Multiply for dark areas
        result[mask] = (2 * accumulator[mask] * current_img[mask]) / 255.0
        
        # Screen for light areas
        result[~mask] = 255.0 - (2 * (255.0 - accumulator[~mask]) * (255.0 - current_img[~mask])) / 255.0
        
        return result
    
    else:
        # Default to lighten
        return np.maximum(accumulator, current_img)


def process_timestack_sliding_window(input_dir, output_dir, file_pattern="*.jpg", 
                                    scale_factor=0.5, max_frames_visible=30,
                                    render_video_flag=True, fps=30, video_quality="high", 
                                    blend_mode="lighten", enable_alignment=True, 
                                    batch_size=50, use_cache=False, simplified_alignment=False,
                                    alignment_scale_factor=0.5, enable_auto_crop=True):
    """
    Create sliding window timestack from image sequence with memory optimizations.
    
    This implementation uses a sliding window approach where only the most recent
    max_frames_visible frames contribute to the final image. When a new frame is added,
    the oldest frame is removed from the accumulator.
    
    Args:
        input_dir: Directory containing source images
        output_dir: Directory for output frames
        file_pattern: Pattern to match image files
        scale_factor: Resize factor for final processing
        max_frames_visible: Maximum frames in sliding window
        render_video_flag: Whether to render video after processing
        fps: Frames per second for video
        video_quality: high, medium, low
        blend_mode: Blending mode to use
        enable_alignment: Whether to align images before blending
        batch_size: Batch size for alignment processing
        use_cache: Use disk cache for aligned images to save memory
        simplified_alignment: Use translation-only alignment for speed
        alignment_scale_factor: Scale factor for feature detection during alignment
        enable_auto_crop: Whether to auto-crop aligned images
    """
    
    # Get sorted list of image files
    image_files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
    
    if not image_files:
        print(f"No images found in {input_dir} matching {file_pattern}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Processing with sliding window of {max_frames_visible} frames")
    print(f"Alignment: {'Enabled' if enable_alignment else 'Disabled'}")
    if enable_alignment:
        print(f"  - Simplified mode: {'Yes' if simplified_alignment else 'No'}")
        print(f"  - Alignment scale factor: {alignment_scale_factor}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Use cache: {'Yes' if use_cache else 'No'}")
        print(f"  - Auto-crop: {'Yes' if enable_auto_crop else 'No'}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize sliding window buffer
    window_buffer = SlidingWindowBuffer(
        max_size=max_frames_visible, 
        use_cache=use_cache,
        cache_dir=os.path.join(output_dir, MEMORY_CACHE_DIR)
    )
    
    # For alignment batch processing
    alignment_buffer = []
    crop_bounds = None
    accumulator = None
    
    try:
        for i, img_path in enumerate(image_files):
            print(f"Processing frame {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load {img_path}")
                continue
            
            # Resize for final processing
            if scale_factor != 1.0:
                height, width = img.shape[:2]
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = cv2.resize(img, (new_width, new_height))
            
            # Handle alignment
            aligned_img = img
            if enable_alignment:
                alignment_buffer.append(img)
                
                # Process in batches for memory efficiency
                if len(alignment_buffer) >= batch_size or i == len(image_files) - 1:
                    print(f"  Aligning batch of {len(alignment_buffer)} images...")
                    
                    try:
                        aligned_batch, transform_matrices = auto_align_images_batch(
                            alignment_buffer,
                            scale_factor=alignment_scale_factor,
                            simplified_mode=simplified_alignment
                        )
                        
                        # Auto-crop batch if enabled and this is the first batch
                        if enable_auto_crop and crop_bounds is None and len(aligned_batch) > 1:
                            print(f"  Computing crop bounds for consistent dimensions...")
                            cropped_batch, crop_bounds = auto_crop_images(aligned_batch, transform_matrices)
                            aligned_batch = cropped_batch
                            print(f"  Crop bounds: {crop_bounds['width']}x{crop_bounds['height']} at ({crop_bounds['x']}, {crop_bounds['y']})")
                        elif enable_auto_crop and crop_bounds is not None:
                            # Apply existing crop bounds
                            cropped_batch = []
                            for img_to_crop in aligned_batch:
                                crop_x = crop_bounds['x']
                                crop_y = crop_bounds['y']
                                crop_w = crop_bounds['width']
                                crop_h = crop_bounds['height']
                                
                                # Ensure crop coordinates are within image bounds
                                height, width = img_to_crop.shape[:2]
                                crop_x_safe = max(0, min(crop_x, width - 1))
                                crop_y_safe = max(0, min(crop_y, height - 1))
                                crop_x2_safe = max(crop_x_safe + 1, min(crop_x + crop_w, width))
                                crop_y2_safe = max(crop_y_safe + 1, min(crop_y + crop_h, height))
                                
                                cropped_img = img_to_crop[crop_y_safe:crop_y2_safe, crop_x_safe:crop_x2_safe]
                                cropped_batch.append(cropped_img)
                            aligned_batch = cropped_batch
                        
                        # Process the last image from the batch
                        aligned_img = aligned_batch[-1]
                        alignment_buffer = []  # Clear buffer after processing
                        
                    except Exception as e:
                        print(f"  Alignment failed: {e}")
                        print(f"  Using original image without alignment")
                        aligned_img = img
                        alignment_buffer = []
                else:
                    # Still collecting images for batch, skip to next
                    continue
            
            # Add to sliding window
            window_buffer.add_image(aligned_img, i)
            
            # Get all images in current window
            window_images = window_buffer.get_all_images()
            
            if not window_images:
                continue
            
            # Recompute accumulator from scratch with current window
            # This ensures proper sliding window behavior
            accumulator = None
            for window_img in window_images:
                if accumulator is None:
                    accumulator = window_img.astype(np.float32)
                else:
                    accumulator = apply_blend_mode(accumulator, window_img.astype(np.float32), blend_mode)
            
            # Save output frame
            if accumulator is not None:
                output_frame = accumulator.astype(np.uint8)
                output_path = os.path.join(output_dir, f"frame_{i+1:04d}.jpg")
                cv2.imwrite(output_path, output_frame)
                
                # Log window status
                if i < max_frames_visible - 1:
                    print(f"  Window: {window_buffer.size()}/{max_frames_visible} frames (building up)")
                else:
                    print(f"  Window: {max_frames_visible} frames (sliding)")
    
    finally:
        # Cleanup
        window_buffer.cleanup()
    
    print(f"\nCompleted! {len(image_files)} frames saved to {output_dir}")
    print(f"Sliding window maintained {max_frames_visible} most recent frames")
    
    # Render video if requested
    if render_video_flag:
        video_output = f"{output_dir}_sliding_timestack{'_aligned' if enable_alignment else ''}.mp4"
        render_video(output_dir, video_output, fps, quality=video_quality)
    else:
        print(f"To create video: ffmpeg -r {fps} -i {output_dir}/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4")

def render_video(frames_dir, output_video, fps=30, codec="libx264", quality="high"):
    """
    Render frames to video using ffmpeg
    
    Args:
        frames_dir: Directory containing frame images
        output_video: Output video filename
        fps: Frames per second
        codec: Video codec (libx264, prores_ks, libx265)
        quality: high, medium, low
    """
    
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg not found. Install ffmpeg to enable video rendering.")
        return False
    
    # Quality settings
    quality_settings = {
        "high": ["-crf", "18"],
        "medium": ["-crf", "23"], 
        "low": ["-crf", "28"]
    }
    
    # Build ffmpeg command
    input_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
    
    cmd = [
        "ffmpeg", "-y",  # Overwrite output
        "-r", str(fps),
        "-i", input_pattern,
        "-c:v", codec,
        "-pix_fmt", "yuv420p"
    ]
    
    # Add quality settings for h264/h265
    if codec in ["libx264", "libx265"]:
        cmd.extend(quality_settings[quality])
    
    cmd.append(output_video)
    
    print(f"Rendering video with ffmpeg...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Video rendered successfully: {output_video}")
            return True
        else:
            print(f"ffmpeg error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error running ffmpeg: {e}")
        return False




def process_raw_timestack(input_dir, output_dir, scale_factor=1.0,
                         render_video_flag=True, fps=24, video_quality="high", blend_mode="lighten"):
    """
    Process RAW files using rawpy (requires: pip install rawpy)
    """
    try:
        import rawpy
    except ImportError:
        print("rawpy not installed. Install with: pip install rawpy")
        return
    
    raw_files = sorted(glob.glob(os.path.join(input_dir, "*.CR2")) + 
                      glob.glob(os.path.join(input_dir, "*.NEF")) + 
                      glob.glob(os.path.join(input_dir, "*.ARW")))
    
    if not raw_files:
        print("No RAW files found")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    accumulator = None
    
    for i, raw_path in enumerate(raw_files):
        print(f"Processing RAW {i+1}/{len(raw_files)}: {os.path.basename(raw_path)}")
        
        with rawpy.imread(raw_path) as raw:
            # Basic RAW processing
            rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Scale down for performance
            if scale_factor != 1.0:
                height, width = bgr.shape[:2]
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                bgr = cv2.resize(bgr, (new_width, new_height))
            
            if accumulator is None:
                accumulator = bgr.astype(np.float32)
            else:
                accumulator = apply_blend_mode(accumulator, bgr.astype(np.float32), blend_mode)
            
            # Save frame
            output_frame = accumulator.astype(np.uint8)
            output_path = os.path.join(output_dir, f"frame_{i+1:04d}.jpg")
            cv2.imwrite(output_path, output_frame)
    
    print(f"\nRAW processing complete! {len(raw_files)} frames saved to {output_dir}")
    
    # Render video if requested
    if render_video_flag:
        video_output = f"{output_dir}_raw_timestack.mp4"
        render_video(output_dir, video_output, fps, quality=video_quality)


def is_video_file(file_path):
    """
    Check if the given file is a video file based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if it's a video file, False otherwise
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    _, ext = os.path.splitext(file_path.lower())
    return ext in video_extensions

def extract_frames_from_video(video_path, output_dir=None):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames (default: based on video name)
        
    Returns:
        str: Path to the directory containing extracted frames
    """
    # Get video file name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = f"{video_name}_frameextract"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting frames from {video_path} to {output_dir}...")
    
    # Use ffmpeg to extract frames
    try:
        # First get video info to determine frame count and fps
        cmd_info = [
            "ffmpeg",
            "-i", video_path,
            "-hide_banner"
        ]
        
        result = subprocess.run(cmd_info, capture_output=True, text=True)
        
        # Extract fps from ffmpeg output
        fps = 30.0  # Default fps if not detected
        for line in result.stderr.split('\n'):
            if "fps" in line and "Stream" in line:
                fps_match = line.split(',')
                for part in fps_match:
                    if "fps" in part:
                        try:
                            fps = float(part.strip().split(' ')[0])
                            break
                        except:
                            pass
                break
        
        print(f"Detected video at {fps} fps")
        
        # Extract frames
        cmd_extract = [
            "ffmpeg",
            "-i", video_path,
            "-q:v", "1",  # High quality
            os.path.join(output_dir, "frame_%04d.jpg")
        ]
        
        print(f"Running command: {' '.join(cmd_extract)}")
        subprocess.run(cmd_extract, check=True)
        
        # Count extracted frames
        extracted_frames = glob.glob(os.path.join(output_dir, "*.jpg"))
        print(f"Extracted {len(extracted_frames)} frames from {video_path}")
        
        return output_dir
    
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sliding window timestack animations with optimizations for large sequences")
    
    # Required arguments
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("output_dir", nargs="?", default="timestack_frames", 
                       help="Output directory for frames (default: timestack_frames)")
    
    # Blending mode options
    parser.add_argument("-b", "--blend", choices=[
        "lighten", "darken", "lighter_color", "darker_color", 
        "difference", "multiply", "screen", "overlay"
    ], default="lighten", help="Blending mode (default: lighten)")
    
    # Processing options - changed default scale factor from 0.5 to 1.0
    parser.add_argument("-s", "--scale", type=float, default=1.0,
                       help="Scale factor for final processing (default: 1.0)")
    
    # Add new sliding window flags
    parser.add_argument("--slide", action="store_true",
                       help="Activate sliding window mode (only most recent frames contribute)")
    parser.add_argument("--swl", "--slide-window-length", type=int, default=30, dest="window_size",
                       help="Sliding window length (default: 30)")
    
    # Existing window size flag (kept for backward compatibility)
    parser.add_argument("-w", "--window-size", type=int, default=30, dest="window_size_legacy",
                       help="Sliding window size (default: 30)")
    
    parser.add_argument("--no-video", action="store_true",
                       help="Don't render video, only create frames")
    parser.add_argument("-f", "--fps", type=int, default=30,
                       help="Video framerate (default: 30)")
    parser.add_argument("-q", "--quality", choices=["high", "medium", "low"],
                       default="high", help="Video quality (default: high)")
    
    # Alignment and performance options
    parser.add_argument("--align", action="store_true",
                       help="Enable image alignment using feature detection")
    parser.add_argument("--simplified-align", action="store_true",
                       help="Use translation-only alignment for faster processing")
    parser.add_argument("--align-scale", type=float, default=0.5,
                       help="Scale factor for feature detection during alignment (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for alignment processing (default: 50)")
    parser.add_argument("--use-cache", action="store_true",
                       help="Cache aligned images to disk if memory is constrained")
    parser.add_argument("--no-auto-crop", action="store_true",
                       help="Disable automatic cropping after alignment")
    
    # File type override
    parser.add_argument("-t", "--type", choices=["auto", "jpg", "png", "raw"],
                       default="auto", help="Force file type (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Extract arguments
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    BLEND_MODE = args.blend
    SCALE_FACTOR = args.scale
    
    # Handle window size parameter with priority to --swl if specified
    if args.window_size != 30:  # User specified --swl
        WINDOW_SIZE = args.window_size
    else:  # Fall back to legacy window size if specified
        WINDOW_SIZE = args.window_size_legacy
    
    # Use sliding window mode if --slide flag is provided
    SLIDING_WINDOW_MODE = args.slide
    
    RENDER_VIDEO = not args.no_video
    FPS = args.fps
    VIDEO_QUALITY = args.quality
    FILE_TYPE = args.type
    ENABLE_ALIGNMENT = args.align
    SIMPLIFIED_ALIGNMENT = args.simplified_align
    ALIGNMENT_SCALE_FACTOR = args.align_scale
    BATCH_SIZE = args.batch_size
    USE_CACHE = args.use_cache
    ENABLE_AUTO_CROP = not args.no_auto_crop
    
    # Print attribution message with first line bold and link blue
    print("\033[1m" + "StackGlider is proudly provided by Benedict Hempel." + "\033[0m")
    print("\033[34m" + "https://github.com/BenedictHempel/timestack_blend_animation" + "\033[0m")
    print()
    
    # Check if input is a video file instead of a directory
    if os.path.isfile(INPUT_DIR) and is_video_file(INPUT_DIR):
        print(f"Detected video file: {INPUT_DIR}")
        frames_dir = extract_frames_from_video(INPUT_DIR)
        if frames_dir:
            print(f"Using extracted frames from: {frames_dir}")
            INPUT_DIR = frames_dir
        else:
            print("Failed to extract frames from video. Exiting.")
            exit(1)
    
    print(f"Processing images from: {INPUT_DIR}")
    print(f"Output frames to: {OUTPUT_DIR}")
    print(f"Window mode: {'Sliding window' if SLIDING_WINDOW_MODE else 'Cumulative (all frames)'}")
    if SLIDING_WINDOW_MODE:
        print(f"Sliding window size: {WINDOW_SIZE} frames")
    print(f"Blend mode: {BLEND_MODE}")
    print(f"Scale factor: {SCALE_FACTOR}")
    print(f"Image alignment: {'Enabled' if ENABLE_ALIGNMENT else 'Disabled'}")
    if ENABLE_ALIGNMENT:
        print(f"  - Simplified alignment: {'Yes' if SIMPLIFIED_ALIGNMENT else 'No'}")
        print(f"  - Alignment scale factor: {ALIGNMENT_SCALE_FACTOR}")
        print(f"  - Batch size: {BATCH_SIZE}")
        print(f"  - Use cache: {'Yes' if USE_CACHE else 'No'}")
        print(f"  - Auto-crop: {'Yes' if ENABLE_AUTO_CROP else 'No'}")
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist!")
        exit(1)
    
    # Auto-detect or force file type
    if FILE_TYPE == "auto":
        jpg_files = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
        png_files = glob.glob(os.path.join(INPUT_DIR, "*.png"))
        raw_files = glob.glob(os.path.join(INPUT_DIR, "*.cr2")) + glob.glob(os.path.join(INPUT_DIR, "*.nef")) + glob.glob(os.path.join(INPUT_DIR, "*.arw"))
        
        if raw_files:
            FILE_TYPE = "raw"
            file_count = len(raw_files)
        elif jpg_files:
            FILE_TYPE = "jpg"
            file_count = len(jpg_files)
        elif png_files:
            FILE_TYPE = "png"
            file_count = len(png_files)
        else:
            print("No supported image files found in input directory!")
            print("Supported formats: .jpg, .jpeg, .png, .cr2, .nef, .arw")
            exit(1)
    else:
        # Count files for forced type
        if FILE_TYPE == "jpg":
            file_count = len(glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.jpeg")))
        elif FILE_TYPE == "png":
            file_count = len(glob.glob(os.path.join(INPUT_DIR, "*.png")))
        elif FILE_TYPE == "raw":
            file_count = len(glob.glob(os.path.join(INPUT_DIR, "*.cr2")) + glob.glob(os.path.join(INPUT_DIR, "*.nef")) + glob.glob(os.path.join(INPUT_DIR, "*.arw")))
        
        if file_count == 0:
            print(f"No {FILE_TYPE.upper()} files found in input directory!")
            exit(1)
    
    print(f"Found {file_count} {FILE_TYPE.upper()} files")
    
    # Ask for confirmation if there are more than 1000 frames
    if file_count > 1000:
        print(f"WARNING: You are about to process {file_count} frames, which may take a long time.")
        confirm = input("Do you want to proceed? (y/n): ").lower()
        if confirm != "y" and confirm != "yes":
            print("Operation cancelled by user.")
            exit(0)
    
    # Process based on file type and mode
    if FILE_TYPE == "raw":
        print("Processing as RAW files...")
        process_raw_timestack(INPUT_DIR, OUTPUT_DIR, scale_factor=SCALE_FACTOR,
                             render_video_flag=RENDER_VIDEO, fps=FPS, 
                             video_quality=VIDEO_QUALITY, blend_mode=BLEND_MODE)
    elif FILE_TYPE in ["jpg", "png"]:
        # Define file pattern based on file type
        file_pattern = "*.jpg" if FILE_TYPE == "jpg" else "*.png"
        
        if SLIDING_WINDOW_MODE:
            print(f"Processing as {FILE_TYPE.upper()} files with sliding window...")
            process_timestack_sliding_window(
                INPUT_DIR, OUTPUT_DIR, file_pattern,
                scale_factor=SCALE_FACTOR,
                max_frames_visible=WINDOW_SIZE,
                render_video_flag=RENDER_VIDEO, 
                fps=FPS, 
                video_quality=VIDEO_QUALITY, 
                blend_mode=BLEND_MODE,
                enable_alignment=ENABLE_ALIGNMENT,
                batch_size=BATCH_SIZE,
                use_cache=USE_CACHE,
                simplified_alignment=SIMPLIFIED_ALIGNMENT,
                alignment_scale_factor=ALIGNMENT_SCALE_FACTOR,
                enable_auto_crop=ENABLE_AUTO_CROP
            )
        else:
            # Add code to handle cumulative (non-sliding) processing mode
            print(f"Processing as {FILE_TYPE.upper()} files with cumulative blending (all frames)...")
            
            # Create output directory
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            
            # Get sorted list of image files
            image_files = sorted(glob.glob(os.path.join(INPUT_DIR, file_pattern)))
            accumulator = None
            
            for i, img_path in enumerate(image_files):
                print(f"Processing frame {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
                
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load {img_path}")
                    continue
                
                # Resize for final processing
                if SCALE_FACTOR != 1.0:
                    height, width = img.shape[:2]
                    new_width = int(width * SCALE_FACTOR)
                    new_height = int(height * SCALE_FACTOR)
                    img = cv2.resize(img, (new_width, new_height))
                
                # Initialize accumulator with first image
                if accumulator is None:
                    accumulator = img.astype(np.float32)
                else:
                    # Apply selected blend mode
                    accumulator = apply_blend_mode(accumulator, img.astype(np.float32), BLEND_MODE)
                
                # Convert back to uint8 and save frame
                output_frame = accumulator.astype(np.uint8)
                output_path = os.path.join(OUTPUT_DIR, f"frame_{i+1:04d}.jpg")
                cv2.imwrite(output_path, output_frame)
            
            print(f"\nCompleted! {len(image_files)} frames saved to {OUTPUT_DIR}")
            
            # Render video if requested
            if RENDER_VIDEO:
                video_output = f"{OUTPUT_DIR}_cumulative_timestack.mp4"
                render_video(OUTPUT_DIR, video_output, FPS, quality=VIDEO_QUALITY)
            else:
                print(f"To create video: ffmpeg -r {FPS} -i {OUTPUT_DIR}/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4")

