import cv2
import numpy as np


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


def compare_keyframes_orb(trailer_keyframes, teaser_keyframes, orb_threshold=180):
    total_teaser_frames = len(teaser_keyframes)
    matched_trailer_frames = 0
    match_details = []  # Store details of matched frames (trailer_frame_index, teaser_frame_index, match_count)

    for i, trailer_frame in enumerate(trailer_keyframes):
        best_match_teaser_index = -1
        max_orb_matches = 0

        for j, teaser_frame in enumerate(teaser_keyframes):
            orb_matches = compare_orb(trailer_frame, teaser_frame)
            if orb_matches > max_orb_matches:
                max_orb_matches = orb_matches
                best_match_teaser_index = j  # Store index of the best matching teaser frame

        if max_orb_matches > orb_threshold:
            matched_trailer_frames += 1
            match_details.append((i, best_match_teaser_index, max_orb_matches))  # Store match details

    return total_teaser_frames, matched_trailer_frames, match_details


def visualize_matched_frames(trailer_keyframes, teaser_keyframes, match_details):
    """
    Display matched frames side by side using OpenCV.
    """
    for trailer_index, teaser_index, match_count in match_details:
        trailer_frame = trailer_keyframes[trailer_index]
        teaser_frame = teaser_keyframes[teaser_index]

        # Resize frames to have the same size (if required)
        if trailer_frame.shape != teaser_frame.shape:
            height = min(trailer_frame.shape[0], teaser_frame.shape[0])
            width = min(trailer_frame.shape[1], teaser_frame.shape[1])
            trailer_frame = cv2.resize(trailer_frame, (width, height))
            teaser_frame = cv2.resize(teaser_frame, (width, height))

        # Stack frames horizontally
        combined_frame = np.hstack((trailer_frame, teaser_frame))
        cv2.putText(combined_frame, f"Matches: {match_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow(f"Matched Frame: Trailer {trailer_index} - Teaser {teaser_index}", combined_frame)
        
        # Wait for key press to move to the next frame (press any key to continue)
        key = cv2.waitKey(0)  # Press any key to view the next frame
        if key == 27:  # If 'Esc' is pressed, exit early
            break
    
    cv2.destroyAllWindows()


def video_correlation(teaser_keyframes, trailer_keyframes):

    print(f"\nTotal trailer keyframes: {len(trailer_keyframes)}")
    print(f"Total teaser keyframes: {len(teaser_keyframes)}")

    print("\nComparing keyframes from trailer and teaser...")
    total_teaser_frames, matched_trailer_frames, match_details = compare_keyframes_orb(
        trailer_keyframes, teaser_keyframes, orb_threshold=180
    )

    if total_teaser_frames > 0:
        match_percentage = (matched_trailer_frames / total_teaser_frames) * 100
    else:
        match_percentage = 0

    print(f"\nMatched Frames: {matched_trailer_frames} out of {total_teaser_frames}")
    print(f"Match Percentage: {match_percentage:.2f}% of teaser keyframes have a strong match in the trailer.\n")

    print(" **Matched Frames**:")
    for trailer_index, teaser_index, match_count in match_details:
        print(f" Trailer Frame {trailer_index} ➡️ Teaser Frame {teaser_index} (ORB Matches: {match_count})")
    
    # Visualize the matched frames
    visualize_matched_frames(trailer_keyframes, teaser_keyframes, match_details)
