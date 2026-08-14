from segmentation import extract_face_regions
from lagenda_face_crop import crop_face

first_image = crop_face(r"C:\Users\imanj\Desktop\lag_benchmark\0b70c012d72b9e97.jpg",238,152,306,230)
# first_image = crop_face(r"C:\Users\imanj\Desktop\lag_benchmark\0b70c012d72b9e97.jpg",920,189,993,272)
#first_image = crop_face(r"C:\Users\imanj\Desktop\lag_benchmark\0b70c012d72b9e97.jpg",724,246,790,323)
first_image.show()
extract_face_regions(first_image)