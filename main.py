from character_detection import detect_celebrities_in_video
from time_duration import get_video_duration
from correlation import video_correlation

if __name__ == "__main__":

    teaser_path = "Videos/Chak De India_Teaser.mp4"
    trailer_path = "Videos/Chak De India Trailer.mp4"
    embeddings_path = "/Users/kirtirane/Downloads/new_data_celebrity_embeddings.npy"
    output_dir = "./output_frames"


    detect_celebrities_in_video(teaser_path, trailer_path, embeddings_path,output_dir)

    trailer_duration = get_video_duration(trailer_path)
    if trailer_duration:
        print(f"Trailer Video Duration: {trailer_duration:.2f} seconds")

    teaser_duration = get_video_duration(teaser_path)
    if teaser_duration:
        print(f"Teaser Video Duration: {teaser_duration:.2f} seconds")

    video_correlation(teaser_path,trailer_path)

    
