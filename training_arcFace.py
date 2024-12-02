import numpy as np
import cv2
import os
from insightface.app import FaceAnalysis


# Initialize ArcFace
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(224, 224))


# Path to celebrity images
celebrity_dataset_dir = '/content/drive/MyDrive/dataset (1)'
celebrity_embeddings = {}


# Function to generate embeddings
def generate_embeddings(image_path):
  
   image = cv2.imread(image_path)
   if image is None:
       print(f"Warning: Unable to read image at {image_path}")
       return None
   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   faces = app.get(image_rgb)  # Detect faces and extract embeddings
   if faces:
       return faces[0].embedding  # Return the embedding of the first face detected
   return None


# Process all images in subfolders
def process_celebrities(celebrity_dataset_dir):
  
   # Valid image file extensions
   valid_extensions = ('.jpg', '.jpeg', '.png')


   for celebrity_folder in os.listdir(celebrity_dataset_dir):
       folder_path = os.path.join(celebrity_dataset_dir, celebrity_folder)


       if os.path.isdir(folder_path):  # Ensure it's a directory
           print(f"Processing folder: {celebrity_folder}")


           for image_file in os.listdir(folder_path):
               image_path = os.path.join(folder_path, image_file)


               # Process only valid image files
               if image_file.lower().endswith(valid_extensions):
                   embedding = generate_embeddings(image_path)
                   if embedding is not None:
                       # Save embedding with a unique key: celebrity_name + image_filename
                       key = f"{celebrity_folder}_{image_file}"
                       celebrity_embeddings[key] = embedding
                   else:
                       print(f"No embeddings generated for {image_file}")
               else:
                   print(f"Skipped: {image_file} is not a valid image file.")


# Main script
if __name__ == "__main__":
   # Process the dataset to generate embeddings
   process_celebrities(celebrity_dataset_dir)


   # Save embeddings to a .npy file for reuse
   output_path = '/content/drive/MyDrive/new_data_celebrity_embeddings.npy'
   np.save(output_path, celebrity_embeddings)
   print(f"Embeddings saved successfully to {output_path}")
