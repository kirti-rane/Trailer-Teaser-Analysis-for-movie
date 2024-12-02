import cv2

def get_video_duration(video_path):
  
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
       print(f"Error: Unable to open video file {video_path}")
       return None

   # Get total frame count and FPS
   frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
   fps = cap.get(cv2.CAP_PROP_FPS)

   cap.release()

   if fps > 0:
       return frame_count / fps
   else:
       print(f"Error: Unable to calculate FPS for {video_path}")
       return None


if __name__ == "__main__":
   # Input video paths with double slashes
   trailer_video_path = "/Users/kirtirane/Downloads/Karthik Calling Karthik- Theatrical Trailer Exclusive!!! - Excel Movies (480p, h264, youtube).mp4"
   teaser_video_path = "/Users/kirtirane/Downloads/Karthik Calling Karthik Teaser - Excel Movies (480p, h264, youtube).mp4"

   # Calculate and display durations
   trailer_duration = get_video_duration(trailer_video_path)
   if trailer_duration:
       print(f"Trailer Video Duration: {trailer_duration:.2f} seconds")

   teaser_duration = get_video_duration(teaser_video_path)
   if teaser_duration:
       print(f"Teaser Video Duration: {teaser_duration:.2f} seconds")

