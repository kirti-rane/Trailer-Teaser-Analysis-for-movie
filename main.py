from character_detection import detect_celebrities_in_video
from time_duration import get_video_duration


if __name__ == "__main__":

    teaser_path = "Videos/Bhool Bhulaiyaa 3 (Teaser)_ Kartik Aaryan, Vidya Balan, Triptii Dimri | Anees Bazmee | Bhushan Kumar - T-Series (720p, h264, youtube).mp4"
    trailer_path = "Videos/Bhool Bhulaiyaa 3 (Official Trailer)_ Kartik Aaryan,Vidya B,Madhuri D,Triptii | Anees B | Bhushan K - T-Series (720p, h264, youtube).mp4"
    embeddings_path = "/Users/kirtirane/Downloads/new_data_celebrity_embeddings.npy"
    output_dir = "./output_frames"


    detect_celebrities_in_video(teaser_path, trailer_path, embeddings_path,output_dir)

    trailer_duration = get_video_duration(trailer_path)
    if trailer_duration:
        print(f"Trailer Video Duration: {trailer_duration:.2f} seconds")

    teaser_duration = get_video_duration(teaser_path)
    if teaser_duration:
        print(f"Teaser Video Duration: {teaser_duration:.2f} seconds")

    
