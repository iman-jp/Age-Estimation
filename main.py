import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Point to your combined images folder
data_dir = 'utkface_images/'

# 2. Grab only the actual JPEG images (ignoring .DS_Store or other files)
all_images = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

#Pick 9 random images to display
sample_images = random.sample(all_images, 9)

# Dictionary mappings for UTKFace labels to make text readable
gender_map = {0: 'Male', 1: 'Female'}
race_map = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}

# Set up a 3x3 grid using Matplotlib
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.ravel() # Flatten the 3x3 grid into a simple 1D array of 9 slots

for i, img_name in enumerate(sample_images):
    # Construct full path and open the image
    img_path = os.path.join(data_dir, img_name)
    img = Image.open(img_path)
    
    parts = img_name.split('_')
    try:
        age = parts[0]
        gender = gender_map[int(parts[1])]
        race = race_map[int(parts[2])]
        title_text = f"Age: {age}\n{gender} | {race}"
    except (IndexError, ValueError):
        title_text = "Unknown Labels"

    # Display the image on the grid
    axes[i].imshow(img)
    axes[i].set_title(title_text, fontsize=10)
    axes[i].axis('off') # Hide the X and Y axes ticks for a cleaner look

#  Adjust layout and show the plot
plt.tight_layout()
plt.show()

