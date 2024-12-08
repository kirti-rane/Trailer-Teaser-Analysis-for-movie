import cv2
import numpy as np
from keyframe_extractor import extract_keyframes  # Import the keyframe extraction logic


def compare_orb(image1, image2):
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    
    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return len(matches)


def compare_keyframes_orb(trailer_keyframes, teaser_keyframes, orb_threshold=80):
    
    total_trailer_frames = len(trailer_keyframes)
    matched_trailer_frames = 0
    match_details = []  # Store details of matched frames (trailer_frame_index, teaser_frame_index, match_count)

    # Compare each trailer keyframe with each teaser keyframe
    for i, trailer_frame in enumerate(trailer_keyframes):
        best_match_teaser_index = -1
        max_orb_matches = 0

        for j, teaser_frame in enumerate(teaser_keyframes):
            orb_matches = compare_orb(trailer_frame, teaser_frame)
            if orb_matches > max_orb_matches:
                max_orb_matches = orb_matches
                best_match_teaser_index = j  # Store index of the best matching teaser frame

        # Check if the frame has a strong match in the teaser
        if max_orb_matches > orb_threshold:
            matched_trailer_frames += 1
            match_details.append((i, best_match_teaser_index, max_orb_matches))  # Store match details

    return total_trailer_frames, matched_trailer_frames, match_details


if __name__ == "__main__":
    trailer_video_path = "Videos/Chak De India Trailer.mp4"
    teaser_video_path = "Videos/Chak De India_Teaser.mp4"

    # Extract keyframes from both trailer and teaser
    print("\n Extracting keyframes from trailer and teaser...")
    trailer_keyframes = extract_keyframes(trailer_video_path, interval=3)
    teaser_keyframes = extract_keyframes(teaser_video_path, interval=3)

    print(f"\nTotal trailer keyframes: {len(trailer_keyframes)}")
    print(f"Total teaser keyframes: {len(teaser_keyframes)}")

    # Compare keyframes using ORB and calculate match percentage
    print("\nComparing keyframes from trailer and teaser...")
    total_trailer_frames, matched_trailer_frames, match_details = compare_keyframes_orb(
        trailer_keyframes, teaser_keyframes, orb_threshold=80
    )

    # Calculate match percentage
    if total_trailer_frames > 0:
        match_percentage = (matched_trailer_frames / total_trailer_frames) * 100
    else:
        match_percentage = 0

    print(f"\nMatched Frames: {matched_trailer_frames} out of {total_trailer_frames}")
    print(f" Match Percentage: {match_percentage:.2f}% of trailer keyframes have a strong match in the teaser.\n")

    # Show matched frame indices
    print(" **Matched Frames**:")
    for trailer_index, teaser_index, match_count in match_details:
        print(f" Trailer Frame {trailer_index} ➡️ Teaser Frame {teaser_index} (ORB Matches: {match_count})")
