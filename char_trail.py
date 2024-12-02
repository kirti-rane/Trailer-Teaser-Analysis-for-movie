import os
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

def extract_name_before_second_underscore(name):
    """Extracts the part of the name before the second underscore."""
    parts = name.split('_')
    if len(parts) > 1:
        return '_'.join(parts[:2])  # Return the first two parts before the second underscore
    return name  # If there is no second underscore, return the original name


def initialize_face_analysis():
    """Initializes the ArcFace model for face detection and embedding generation."""
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(224, 224))
    return app


def load_celebrity_embeddings(file_path):
    """Loads celebrity embeddings from a file."""
    return np.load(file_path, allow_pickle=True).item()


def generate_embeddings(image, face_analyzer):
    """Generates face embeddings for all detected faces in an image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_analyzer.get(image_rgb)  # Detect faces and extract embeddings
    if faces:
        return [(face.bbox, face.embedding) for face in faces]
    return []


def annotate_and_save_frame(frame, bbox, label, celebrity_folder, frame_count):
    """Annotates a frame with a bounding box and a label, then saves it."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
   


def process_video(video_path, celebrity_embeddings, face_analyzer, output_dir, threshold=0.5, frame_interval=2):
    """Processes video frames to detect celebrities and annotate frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    results = []
    unique_characters = set()  # To store unique detected celebrities

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % int(fps * frame_interval) == 0:
            face_data = generate_embeddings(frame, face_analyzer)
            if not face_data:
                frame_count += 1
                continue

            for bbox, embedding in face_data:
                best_match = None
                best_similarity = 0
                for celebrity, celeb_embedding in celebrity_embeddings.items():
                    similarity = cosine_similarity([embedding], [celeb_embedding])[0][0]
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_match = celebrity

                if best_match:
                    clean_name = extract_name_before_second_underscore(best_match)
                    label = f"{clean_name} ({best_similarity:.2f})"
                    results.append((frame_count, best_match, best_similarity))
                    unique_characters.add(clean_name)

                    # Save annotated frame to the celebrity's folder
                    celebrity_folder = os.path.join(output_dir, clean_name)
                    annotate_and_save_frame(frame, bbox, label, celebrity_folder, frame_count)

            # Display the frame with annotations
            cv2.imshow('Detected Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return results, list(unique_characters)


if __name__ == "__main__":
    # Paths and Initialization
    teaser_path = "Videos/Bhool Bhulaiyaa 3 (Official Trailer)_ Kartik Aaryan,Vidya B,Madhuri D,Triptii | Anees B | Bhushan K - T-Series (720p, h264, youtube).mp4"
    trailer_path = "Videos/Bhool Bhulaiyaa 3 (Official Trailer)_ Kartik Aaryan,Vidya B,Madhuri D,Triptii | Anees B | Bhushan K - T-Series (720p, h264, youtube).mp4"
    embeddings_path = "/Users/kirtirane/Downloads/new_data_celebrity_embeddings.npy"
    output_dir = "./output_frames"  # Output directory for saved frames

    face_analyzer = initialize_face_analysis()
    celebrity_embeddings = load_celebrity_embeddings(embeddings_path)

    # Process teaser
    print("Processing Teaser...")
    teaser_results, teaser_unique_characters = process_video(teaser_path, celebrity_embeddings, face_analyzer, output_dir)

    # Print teaser results
    print("Teaser Results:")
    if teaser_results:
        for frame_idx, celebrity, similarity in teaser_results:
            print(f"Frame: {frame_idx}, Matched Celebrity: {celebrity}, Similarity: {similarity}")
    else:
        print("No celebrities were detected in the teaser.")

    print("\nUnique characters in teaser:")
    print(teaser_unique_characters)

    # Process trailer
    print("\nProcessing Trailer...")
    trailer_results, trailer_unique_characters = process_video(trailer_path, celebrity_embeddings, face_analyzer, output_dir)

    # Print trailer results
    print("Trailer Results:")
    if trailer_results:
        for frame_idx, celebrity, similarity in trailer_results:
            print(f"Frame: {frame_idx}, Matched Celebrity: {celebrity}, Similarity: {similarity}")
    else:
        print("No celebrities were detected in the trailer.")

    print("\nUnique characters in trailer:")
    print(trailer_unique_characters)

    # Compare teaser and trailer unique characters
    common_characters = set(teaser_unique_characters).intersection(trailer_unique_characters)
    print("\nCommon characters between teaser and trailer:")
    print(list(common_characters))
