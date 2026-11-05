"""
Image Preprocessing Pipeline for Real-World Handwritten Digit Photos
=====================================================================
Transforms any arbitrary user photo (smartphone capture, webcam, scan, canvas drawing)
into the exact 28x28 center-of-mass normalized format expected by the MNIST Neural Network.

MNIST Standard Preprocessing Rules (LeCun et al.):
1. Grayscale representation.
2. Inverted: Bright digit strokes on Dark background (0 = black background, 1 = white stroke).
3. Foreground extraction: Crop tightly around digit strokes.
4. Scale preserving aspect ratio to fit inside a 20x20 bounding box.
5. Center inside a 28x28 canvas using Center of Mass (Centroid).
6. Pixel normalization: [0, 255] -> [0.0, 1.0].
"""

import io
import numpy as np
from PIL import Image, ImageOps, ImageFilter

class DigitImagePreprocessor:
    def __init__(self, target_size=(28, 28), inner_box=20):
        self.target_size = target_size
        self.inner_box = inner_box

    def load_image(self, image_input):
        """Loads image from file path, bytes, or PIL Image."""
        if isinstance(image_input, str):
            return Image.open(image_input)
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def preprocess(self, image_input, return_intermediate=False):
        """
        Main pipeline to preprocess an input photo into MNIST format.
        
        Returns:
            processed_vector: np.ndarray of shape (784,) with values in [0.0, 1.0]
            intermediate_dict (optional): dictionary with step-by-step images for visualization
        """
        orig_img = self.load_image(image_input).convert("RGBA")
        
        # If image has transparency (alpha channel), paste onto white background
        background = Image.new("RGBA", orig_img.size, (255, 255, 255))
        alpha_composite = Image.alpha_composite(background, orig_img)
        gray = alpha_composite.convert("L")

        # Step 1: Auto-detect background brightness and invert if necessary
        # MNIST requires: Bright strokes on Dark background.
        # Paper photos are: Dark strokes on Light background.
        gray_np = np.array(gray)
        # Sample border pixels (top, bottom, left, right edges)
        border_pixels = np.concatenate([
            gray_np[0, :], gray_np[-1, :],
            gray_np[:, 0], gray_np[:, -1]
        ])
        border_mean = np.mean(border_pixels)
        
        # If border is bright (> 127), photo is dark ink on light paper -> Invert!
        if border_mean > 127:
            inverted = ImageOps.invert(gray)
        else:
            inverted = gray

        inv_np = np.array(inverted, dtype=np.float32)

        # Step 2: Adaptive Noise Removal & Contrast Stretching
        # Subtract background noise floor (e.g. 10th percentile of inverted)
        bg_level = np.percentile(inv_np, 20)
        inv_np = np.maximum(0, inv_np - bg_level)
        
        max_val = np.max(inv_np)
        if max_val > 10:
            inv_np = (inv_np / max_val) * 255.0
        
        # Step 3: Thresholding to isolate digit contour
        threshold = 45.0
        inv_np[inv_np < threshold] = 0.0

        # Step 4: Find Bounding Box of the digit
        nonzero_coords = np.argwhere(inv_np > 30)
        if len(nonzero_coords) == 0:
            # Blank image provided
            blank = np.zeros(self.target_size, dtype=np.float32)
            vector = blank.flatten()
            if return_intermediate:
                return vector, {"status": "blank", "final_28x28": blank}
            return vector

        ymin, xmin = nonzero_coords.min(axis=0)
        ymax, xmax = nonzero_coords.max(axis=0) + 1

        digit_crop = inv_np[ymin:ymax, xmin:xmax]
        crop_pil = Image.fromarray(digit_crop.astype(np.uint8))

        # Step 5: Resize preserving aspect ratio to fit inside 20x20 inner box
        w, h = crop_pil.size
        if w > h:
            new_w = self.inner_box
            new_h = max(1, int(round((h * self.inner_box) / w)))
        else:
            new_h = self.inner_box
            new_w = max(1, int(round((w * self.inner_box) / h)))

        resized_digit = crop_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Step 6: Center in 28x28 canvas using Center of Mass (Centroid)
        canvas = np.zeros(self.target_size, dtype=np.float32)
        
        # First place roughly at center
        ox = (self.target_size[1] - new_w) // 2
        oy = (self.target_size[0] - new_h) // 2
        canvas[oy:oy+new_h, ox:ox+new_w] = np.array(resized_digit, dtype=np.float32)

        # Compute Center of Mass
        total_mass = np.sum(canvas)
        if total_mass > 0:
            y_indices, x_indices = np.indices(canvas.shape)
            cy = np.sum(y_indices * canvas) / total_mass
            cx = np.sum(x_indices * canvas) / total_mass

            # Shift so center of mass is at (13.5, 13.5)
            shift_y = int(round(13.5 - cy))
            shift_x = int(round(13.5 - cx))

            # Apply shift using roll and boundary zeroing
            if shift_y != 0 or shift_x != 0:
                shifted = np.roll(canvas, shift_y, axis=0)
                shifted = np.roll(shifted, shift_x, axis=1)
                
                # Zero out wrapped borders
                if shift_y > 0:
                    shifted[:shift_y, :] = 0
                elif shift_y < 0:
                    shifted[shift_y:, :] = 0
                if shift_x > 0:
                    shifted[:, :shift_x] = 0
                elif shift_x < 0:
                    shifted[:, shift_x:] = 0
                canvas = shifted

        # Step 7: Normalize to [0.0, 1.0]
        final_normalized = canvas / 255.0
        final_vector = final_normalized.flatten()

        if return_intermediate:
            intermediates = {
                "original_size": orig_img.size,
                "grayscale": gray,
                "inverted_thresholded": Image.fromarray(inv_np.astype(np.uint8)),
                "cropped_digit": crop_pil,
                "final_28x28": Image.fromarray((canvas).astype(np.uint8)),
                "center_of_mass": (float(cx) if total_mass > 0 else 13.5, float(cy) if total_mass > 0 else 13.5)
            }
            return final_vector, intermediates

        return final_vector

if __name__ == "__main__":
    import sys
    preprocessor = DigitImagePreprocessor()
    test_path = sys.argv[1] if len(sys.argv) > 1 else "sample_images/photo_paper_digit_7.jpg"
    print(f"Testing preprocessor on: {test_path}")
    vec, debug = preprocessor.preprocess(test_path, return_intermediate=True)
    print(f"Preprocessed vector shape: {vec.shape}, min: {vec.min():.3f}, max: {vec.max():.3f}, mean: {vec.mean():.3f}")
    debug["final_28x28"].save("sample_images/debug_preprocessed_28x28.png")
    print("Saved preprocessed debug image to sample_images/debug_preprocessed_28x28.png")

