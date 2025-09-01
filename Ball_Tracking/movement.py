import pyrealsense2 as rs
import numpy as np
import cv2

def start_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    return pipeline, profile

# def get_frames(pipeline):
#     frames = pipeline.wait_for_frames()
#     depth_frame = frames.get_depth_frame()
#     color_frame = frames.get_color_frame()
#     if not depth_frame or not color_frame:
#         return None, None
#     depth_image = np.asanyarray(depth_frame.get_data())
#     color_image = np.asanyarray(color_frame.get_data())
#     return depth_image, color_image

def get_aligned_frames(pipeline):
    align = rs.align(rs.stream.color)  # align depth to color
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_image, color_image

def normalize_depth_for_display(depth_image):
    # Convert depth to 8-bit grayscale for visualization
    depth_colored = cv2.convertScaleAbs(depth_image, alpha=0.03)
    depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)
    return depth_colored

def overlay_motion_depth(prev_frame, curr_frame, depth_image, threshold=30):
    # Compute absolute difference and grayscale it
    diff = cv2.absdiff(curr_frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Create motion mask
    _, motion_mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    motion_mask_colored = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    # Get displayable depth
    depth_display = normalize_depth_for_display(depth_image)

    # Combine RGB + depth using mask
    blended = np.where(motion_mask_colored == 255, depth_display, curr_frame)

    return blended, motion_mask


def overlay_motion_depth(prev_frame, curr_frame, depth_image, threshold=30):
    # Compute absolute difference and grayscale it
    diff = cv2.absdiff(curr_frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Create motion mask
    _, motion_mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    motion_mask_colored = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    # Get displayable depth
    depth_display = normalize_depth_for_display(depth_image)

    # Overlay depth on RGB wherever motion is detected
    blended = np.where(motion_mask_colored == 255, depth_display, curr_frame)

    # Create depth image with white mask overlay for debugging
    mask_overlay_on_depth = depth_display.copy()
    mask_overlay_on_depth[motion_mask == 255] = [255, 255, 255]

    return blended, motion_mask, mask_overlay_on_depth


def run_motion_depth_overlay():
    pipeline, _ = start_camera()
    prev_frame = None

    try:
        while True:
            depth_image, color_image = get_aligned_frames(pipeline)
            if depth_image is None or color_image is None:
                continue

            if prev_frame is None:
                prev_frame = color_image
                continue

            overlayed, motion_mask, debug_depth = overlay_motion_depth(prev_frame, color_image, depth_image)

            # Show the results
            cv2.imshow('Motion Overlay', overlayed)
            cv2.imshow('Motion Mask', motion_mask)
            cv2.imshow('Depth + Mask Overlay (white)', debug_depth)

            prev_frame = color_image.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# Run it
run_motion_depth_overlay()