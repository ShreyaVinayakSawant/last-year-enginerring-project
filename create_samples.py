import os
import numpy as np
import cv2

def generate_sample_images():
    output_dir = "sample_images"
    os.makedirs(output_dir, exist_ok=True)
    
    samples = [
        ("Tomato_Healthy_Sample.jpg", (40, 180, 50), "HEALTHY", (200, 200)),
        ("Tomato_Early_Blight_Sample.jpg", (30, 100, 140), "EARLY BLIGHT", (250, 150)),
        ("Corn_Common_Rust_Sample.jpg", (20, 80, 160), "COMMON RUST", (100, 220)),
        ("Apple_Scab_Sample.jpg", (50, 90, 70), "APPLE SCAB", (180, 180)),
        ("Potato_Late_Blight_Sample.jpg", (40, 60, 50), "LATE BLIGHT", (120, 140))
    ]
    
    for filename, leaf_color, text, spot_color in samples:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            # Create a 256x256 image representing a leaf
            img = np.ones((256, 256, 3), dtype=np.uint8) * 240 # light background
            
            # Draw leaf shape
            center = (128, 128)
            axes = (80, 110)
            angle = 15
            cv2.ellipse(img, center, axes, angle, 0, 360, leaf_color, -1)
            cv2.ellipse(img, center, axes, angle, 0, 360, (20, 70, 20), 3) # border
            
            # Draw stem & veins
            cv2.line(img, (128, 238), (128, 30), (20, 70, 20), 3)
            cv2.line(img, (128, 140), (80, 100), (20, 70, 20), 2)
            cv2.line(img, (128, 140), (176, 100), (20, 70, 20), 2)
            cv2.line(img, (128, 180), (70, 150), (20, 70, 20), 2)
            cv2.line(img, (128, 180), (186, 150), (20, 70, 20), 2)
            
            # Draw spots if not healthy
            if "HEALTHY" not in text:
                for _ in range(8):
                    rx = np.random.randint(80, 176)
                    ry = np.random.randint(80, 176)
                    rsize = np.random.randint(5, 18)
                    cv2.circle(img, (rx, ry), rsize, spot_color, -1)
                    cv2.circle(img, (rx, ry), rsize + 2, (10, 10, 10), 1)

            cv2.imwrite(filepath, img)
            print(f"Generated sample image: {filepath}")

if __name__ == "__main__":
    generate_sample_images()
