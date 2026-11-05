import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Load dataset
npz_path = os.path.join(DATA_DIR, "mnist.npz")
data = np.load(npz_path)
x_test = data["x_test"]
y_test = data["y_test"]

print(f"Loaded {len(x_test)} test images.")

# 1. Save standard MNIST test samples for each digit 0-9
found_digits = {}
for i, (img, label) in enumerate(zip(x_test, y_test)):
    if label not in found_digits:
        found_digits[label] = img
        # Save standard 28x28 sample
        pil_img = Image.fromarray(img)
        # Also upscale to 280x280 for easy human viewing
        upscaled = pil_img.resize((280, 280), Image.NEAREST)
        save_path = os.path.join(SAMPLES_DIR, f"mnist_digit_{label}.png")
        upscaled.save(save_path)
        print(f"Saved MNIST sample digit {label} -> {save_path}")
    if len(found_digits) == 10:
        break

# 2. Create realistic "Photo on Paper" test pictures (simulating smartphone camera photos)
# Real photos typically have:
# - White/off-white notebook paper with faint texture or lighting gradient
# - Dark ink (black, blue, or pencil)
# - Higher resolution (e.g., 600x600)
# - Digit drawn in center
def create_realistic_photo(digit, base_mnist_img, filename):
    size = (600, 600)
    # Background: off-white cream paper (RGB ~ 245, 242, 235) with subtle lighting gradient
    np_paper = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        # Slight vertical lighting gradient
        grad = int(10 * (y / size[1]))
        np_paper[y, :, 0] = 245 - grad
        np_paper[y, :, 1] = 243 - grad
        np_paper[y, :, 2] = 238 - grad
    
    # Add subtle paper noise
    noise = np.random.randint(-4, 5, size=(size[1], size[0], 3))
    np_paper = np.clip(np_paper.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    paper_img = Image.fromarray(np_paper)
    
    # Resize MNIST digit (which is white on black)
    digit_pil = Image.fromarray(base_mnist_img).resize((320, 320), Image.Resampling.BILINEAR)
    digit_pil = digit_pil.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # Convert digit to dark ink mask (alpha)
    digit_arr = np.array(digit_pil)
    ink_mask = digit_arr / 255.0  # 1.0 where stroke is strong
    
    # Create ink color: dark blue/black ballpoint pen (RGB ~ 25, 30, 45)
    ink_color = np.array([25, 30, 50], dtype=np.float32)
    
    # Paste onto paper at center
    ox, oy = (size[0] - 320) // 2, (size[1] - 320) // 2
    paper_region = np_paper[oy:oy+320, ox:ox+320].astype(np.float32)
    
    for c in range(3):
        paper_region[:, :, c] = paper_region[:, :, c] * (1.0 - ink_mask * 0.85) + ink_color[c] * (ink_mask * 0.85)
    
    np_paper[oy:oy+320, ox:ox+320] = np.clip(paper_region, 0, 255).astype(np.uint8)
    
    result = Image.fromarray(np_paper)
    out_path = os.path.join(SAMPLES_DIR, filename)
    result.save(out_path, quality=95)
    print(f"Created realistic smartphone photo of digit {digit} -> {out_path}")

for digit in [0, 2, 3, 5, 7, 8]:
    create_realistic_photo(digit, found_digits[digit], f"photo_paper_digit_{digit}.jpg")

print("\nAll sample pictures created successfully in sample_images/")

