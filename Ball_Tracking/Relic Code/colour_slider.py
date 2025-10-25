import cv2
import numpy as np

def create_hsv_trackbars():
    """Create a window with trackbars for HSV adjustment"""
    cv2.namedWindow('HSV Adjustment')
    
    # Initial values (will be updated after calibration)
    cv2.createTrackbar('H min', 'HSV Adjustment', 0, 179, lambda x: None)
    cv2.createTrackbar('H max', 'HSV Adjustment', 0, 179, lambda x: None)
    cv2.createTrackbar('S min', 'HSV Adjustment', 0, 255, lambda x: None)
    cv2.createTrackbar('S max', 'HSV Adjustment', 0, 255, lambda x: None)
    cv2.createTrackbar('V min', 'HSV Adjustment', 0, 255, lambda x: None)
    cv2.createTrackbar('V max', 'HSV Adjustment', 0, 255, lambda x: None)

def get_hsv_from_trackbars():
    """Get the HSV ranges from trackbars"""
    h_min = cv2.getTrackbarPos('H min', 'HSV Adjustment')
    h_max = cv2.getTrackbarPos('H max', 'HSV Adjustment')
    s_min = cv2.getTrackbarPos('S min', 'HSV Adjustment')
    s_max = cv2.getTrackbarPos('S max', 'HSV Adjustment')
    v_min = cv2.getTrackbarPos('V min', 'HSV Adjustment')
    v_max = cv2.getTrackbarPos('V max', 'HSV Adjustment')
    
    return (h_min, h_max), (s_min, s_max), (v_min, v_max)
    

def update_hsv_trackbars(hsv_range):
    """Update trackbar positions based on calibrated HSV values"""
    
    h_min = max(0,   int(hsv_range[0][0]))
    h_max = min(179, int(hsv_range[0][1]))
    s_min = max(0,   int(hsv_range[1][0]))
    s_max = min(255, int(hsv_range[1][1]))
    v_min = max(0,   int(hsv_range[2][0]))
    v_max = min(255, int(hsv_range[2][1]))
    
    cv2.setTrackbarPos('H min', 'HSV Adjustment', h_min)
    cv2.setTrackbarPos('H max', 'HSV Adjustment', h_max)
    cv2.setTrackbarPos('S min', 'HSV Adjustment', s_min)
    cv2.setTrackbarPos('S max', 'HSV Adjustment', s_max)
    cv2.setTrackbarPos('V min', 'HSV Adjustment', v_min)
    cv2.setTrackbarPos('V max', 'HSV Adjustment', v_max)

def render_hsv_color_visualization(h_range, s_range, v_range):
    """Create a visualization of the selected HSV color range"""
    # Create a visualization pane of size 400x300
    width, height = 400, 300
    visualization = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create H, S, V color gradients
    h_gradient_height = height // 3
    s_gradient_height = height // 3
    v_gradient_height = height // 3
    
    # Create H gradient (horizontal)
    for x in range(width):
        h_val = int(x * 180 / width)  # 0-179
        for y in range(h_gradient_height):
            # Highlight the selected range
            alpha = 1.0 if h_range[0] <= h_val <= h_range[1] else 0.3
            # Full saturation and value for clear hue visualization
            hsv_color = np.array([h_val, 255, 255], dtype=np.uint8)
            bgr_color = cv2.cvtColor(np.array([[hsv_color]]), cv2.COLOR_HSV2BGR)[0, 0]
            # Apply alpha for non-selected areas
            if alpha < 1.0:
                bgr_color = (bgr_color * alpha).astype(np.uint8)
            visualization[y, x] = bgr_color
    
    # Draw H range text
    cv2.putText(visualization, f"H: {h_range[0]}-{h_range[1]}", 
                (10, h_gradient_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Create S gradient (horizontal)
    s_start = h_gradient_height
    h_mid = (h_range[0] + h_range[1]) // 2  # Use middle of H range
    v_mid = (v_range[0] + v_range[1]) // 2  # Use middle of V range
    
    for x in range(width):
        s_val = int(x * 255 / width)  # 0-255
        for y in range(s_start, s_start + s_gradient_height):
            alpha = 1.0 if s_range[0] <= s_val <= s_range[1] else 0.3
            hsv_color = np.array([h_mid, s_val, v_mid], dtype=np.uint8)
            bgr_color = cv2.cvtColor(np.array([[hsv_color]]), cv2.COLOR_HSV2BGR)[0, 0]
            if alpha < 1.0:
                bgr_color = (bgr_color * alpha).astype(np.uint8)
            visualization[y, x] = bgr_color
    
    # Draw S range text
    cv2.putText(visualization, f"S: {s_range[0]}-{s_range[1]}", 
                (10, s_start + s_gradient_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Create V gradient (horizontal)
    v_start = s_start + s_gradient_height
    for x in range(width):
        v_val = int(x * 255 / width)  # 0-255
        for y in range(v_start, v_start + v_gradient_height):
            alpha = 1.0 if v_range[0] <= v_val <= v_range[1] else 0.3
            hsv_color = np.array([h_mid, s_mid := (s_range[0] + s_range[1]) // 2, v_val], dtype=np.uint8)
            bgr_color = cv2.cvtColor(np.array([[hsv_color]]), cv2.COLOR_HSV2BGR)[0, 0]
            if alpha < 1.0:
                bgr_color = (bgr_color * alpha).astype(np.uint8)
            visualization[y, x] = bgr_color
    
    # Draw V range text
    cv2.putText(visualization, f"V: {v_range[0]}-{v_range[1]}", 
                (10, v_start + v_gradient_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Create a color sample showing the selected color range
    sample_height = 40
    sample = np.zeros((sample_height, width, 3), dtype=np.uint8)
    
    # Show actual color range at bottom
    for x in range(width):
        # Vary H across width within selected range
        h_span = h_range[1] - h_range[0]
        h = h_range[0] + int((x / width) * h_span) if h_span > 0 else h_range[0]
        
        for y in range(sample_height):
            # Vary S and V based on position within the sample
            s_span = s_range[1] - s_range[0]
            v_span = v_range[1] - v_range[0]
            
            # Distribute S and V across sample
            s = s_range[0] + int((y / sample_height) * s_span) if s_span > 0 else s_mid
            v = v_range[1] - int((y / sample_height) * v_span) if v_span > 0 else v_mid
            
            hsv_color = np.array([h, s, v], dtype=np.uint8)
            sample[y, x] = cv2.cvtColor(np.array([[hsv_color]]), cv2.COLOR_HSV2BGR)[0, 0]
    
    # Combine visualization and sample
    result = np.vstack([visualization, sample])
    
    return result