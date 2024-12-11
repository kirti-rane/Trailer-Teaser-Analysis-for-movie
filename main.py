from character_detection import detect_celebrities_in_video
from time_duration import get_video_duration
from correlation import video_correlation
from keyframe_extractor import extract_keyframes
import numpy as np


if __name__ == "__main__":

    teaser_path = "Videos/GK_teaser.mp4"
    trailer_path = "Videos/GK_trailer.mp4"
    embeddings_path = "Trained_model/new_data_celebrity_embeddings.npy"
    output_dir = "./output_frames"

    print("\n Extracting keyframes from trailer and teaser...")
    trailer_keyframes = extract_keyframes(trailer_path, interval=1)
    teaser_keyframes = extract_keyframes(teaser_path, interval=1)

    detect_celebrities_in_video(teaser_keyframes, trailer_keyframes, embeddings_path,output_dir)

    trailer_duration = get_video_duration(trailer_path)
    if trailer_duration:
        print(f"Trailer Video Duration: {trailer_duration:.2f} seconds")

    teaser_duration = get_video_duration(teaser_path)
    if teaser_duration:
        print(f"Teaser Video Duration: {teaser_duration:.2f} seconds")

    video_correlation(teaser_keyframes,trailer_keyframes)

    
