import time
import pyrealsense2 as rs
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import colour_slider

# ------ Configurations ------
COLOUR_TOLERANCE = 10  # HSV threshold range
DEPTH_TOLERANCE = 100  # 
SEARCH_RADIUS = 200
DBSCAN_EPS = 50  # mm
DBSCAN_MIN_SAMPLES = 10
BALL_DIAMETER_MM = 40 

# ------ Initial Setup ------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)

profile = pipeline.start(config)

# Align depth to colour
align = rs.align(rs.stream.color)

# Get depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# ------ Variables ------
calibrated = False
ball_hsv = None
ball_depth = None
last_ball_pos = None
flight_tracking = False
flight_path = []

# ------ New Calibration Variables ------
INIT_SEARCH_CIRCLE = 70  # pixels, bigger circle
STABILITY_TIME_REQUIRED = 2.0  # seconds
hough_dp = 1.2
hough_min_dist = 30
hough_param1 = 100
hough_param2 = 30
hough_min_radius = INIT_SEARCH_CIRCLE - 20
hough_max_radius = INIT_SEARCH_CIRCLE + 20

init_ball_radius = 0

calibration_start_time = None
ball_ready = False

hsv_range = [[0, 179], [206, 255], [59, 255]]

def detect_circle_in_centre(color_image):
	gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
	gray = cv2.medianBlur(gray, 5)

	circles = cv2.HoughCircles(
		gray,
		cv2.HOUGH_GRADIENT,
		dp=hough_dp,
		minDist=hough_min_dist,
		param1=hough_param1,
		param2=hough_param2,
		minRadius=hough_min_radius,
		maxRadius=hough_max_radius
	)

	if circles is not None:
		circles = np.uint16(np.around(circles))
		h, w = color_image.shape[:2]
		centre = (w // 2, h // 2)

		for circle in circles[0, :]:
			x, y, r = circle
			dist = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
			if dist < 20:  # must be close to centre
				return True, (x, y), r
	return False, None, None


def extract_ball_colour_and_depth(color_frame, depth_frame, centre):
	mask = np.zeros(color_frame.shape[:2], dtype=np.uint8)
	cv2.circle(mask, centre, INIT_SEARCH_CIRCLE, 255, -1)

	hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
	mean_hsv = cv2.mean(hsv, mask=mask)[:3]

	depth_image = np.asanyarray(depth_frame.get_data())
	depth_values = depth_image[mask == 255]
	valid_depths = depth_values[depth_values > 0]
	mean_depth = np.median(valid_depths) * depth_scale * 1000  # mm

	return mean_hsv, mean_depth

# Update find_ball_local to use DEPTH_TOLERANCE instead of 500
def find_ball_local(depth_img, h_range, s_range, v_range, ball_depth, ball_detected, last_centres):
	
	# Create mask using slider values
	lower = np.array([h_range[0], s_range[0], v_range[0]])
	upper = np.array([h_range[1], s_range[1], v_range[1]])
	
	# Apply color mask
	mask = cv2.inRange(hsv_image, lower, upper)

	# Depth threshold
	depth_in_mm = depth_img * depth_scale * 1000
	depth_mask = np.logical_and(depth_in_mm > (ball_depth - DEPTH_TOLERANCE),
								depth_in_mm < (ball_depth + DEPTH_TOLERANCE))
	depth_mask = depth_mask.astype(np.uint8) * 255

	# combined_mask = depth_mask
	if not ball_detected:
		combined_mask = mask
	else:
		combined_mask = cv2.bitwise_and(mask, depth_mask)

	if len(last_centres) > 0:
		# Limit search area around last known centre
		search_mask = np.zeros_like(combined_mask)
		cx, cy = last_centres[-1]

		cv2.circle(search_mask, (cx, cy), SEARCH_RADIUS, 255, -1)
		combined_mask = cv2.bitwise_and(combined_mask, search_mask)
		
		if len(last_centres) > 1:
			velocity = np.array(last_centres[-1]) - np.array(last_centres[-2])
			velocity_mg = np.linalg.norm(velocity)

			predicted_cent = last_centres[-1] + velocity
			cx2, cy2 = predicted_cent
			cv2.circle(search_mask, (cx2, cy2), SEARCH_RADIUS, 255, -1)

	return combined_mask

def is_ball_in_air(cx, cy, ball_depth, expected_radius, depth_frame):
	# Crop small region around ball
	depth_image = np.asanyarray(depth_frame.get_data())
	h, w = depth_image.shape
	search_radius = expected_radius + 10  # 5 cm extra buffer
	depth_tolerance = 15  # +-0.1m tolerance around the ball's depth

	# Generate pixel coordinates grid
	y, x = np.indices((h, w))
	
	depth_mask = (depth_image > 0) & (abs(depth_image - ball_depth) <= depth_tolerance)
	distance_mask = (x - cx)**2 + (y - cy)**2 <= (search_radius * search_radius)

	# depth_mask_img = np.zeros((h, w, 3), dtype=np.uint8)
	# depth_mask_img[depth_mask] = [0, 255, 0]  # Green for depth mask
	# cv2.imshow("Depth Mask", depth_mask_img)

	# # Debug: Show distance mask
	# distance_mask_img = np.zeros((h, w, 3), dtype=np.uint8)
	# distance_mask_img[distance_mask] = [255, 0, 0]  # Red for distance mask
	# cv2.imshow("Distance Mask", distance_mask_img)

	combined_mask = depth_mask & distance_mask
	valid_points = np.column_stack(np.where(combined_mask)) 
	
	if len(valid_points) < 5:
		return False

	# Perform 2D clustering on the valid points (just x, y)
	clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(valid_points)
	labels = clustering.labels_

	# Get the unique clusters (ignore noise labeled as -1)
	unique_labels = np.unique(labels)
	valid_clusters = [label for label in unique_labels if label != -1]
	
	if len(valid_clusters) == 0:
		return False

	# Select the largest cluster
	largest_cluster = max(valid_clusters, key=lambda label: np.sum(labels == label))

	# Get the points of the largest cluster
	cluster_points = valid_points[labels == largest_cluster]

	# Calculate the bounding box of the cluster in pixel coordinates
	min_x, min_y = np.min(cluster_points, axis=0)
	max_x, max_y = np.max(cluster_points, axis=0)

	# Check if the pixel span in x and y directions is within the expected radius
	cluster_width = max_x - min_x  # Width in pixels
	cluster_height = max_y - min_y  # Height in pixels

	debug_image = cv2.cvtColor(depth_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
	debug_image[valid_points[:, 0], valid_points[:, 1]] = [0, 255, 0]  # Show valid points in green
	cv2.rectangle(debug_image, (min_y, min_x), (max_y, max_x), (0, 0, 255), 2)  # Draw the bounding box in red
	cv2.imshow("Cluster Bounding Box", debug_image)
	
	if cluster_width <= expected_radius * 2 + 0.01 and cluster_height <= expected_radius * 2 + 0.01:
		return True
	else:
		return False

def apply_focus_mask(image, centre, radius):
	mask = np.zeros_like(image, dtype=np.uint8)
	cv2.circle(mask, centre, radius, (255, 255, 255), -1)
	dark_overlay = (image * 0.7).astype(np.uint8)  # Darken whole frame
	focused = np.where(mask==255, image, dark_overlay)  # Keep only circle bright
	return focused

def draw_progress_arc(image, centre, radius, progress, colour=(255, 255, 255), thickness=4):
	# progress is 0.0 to 1.0
	end_angle = int(360 * progress)
	cv2.ellipse(
		image, 
		centre, 
		(radius, radius), 
		0,             # rotation angle
		-90,           # start angle (top)
		-90 + end_angle, 
		colour, 
		thickness, 
		lineType=cv2.LINE_AA
	)

def predict_radius(calibrated_radius, calibrated_depth, current_depth):
    if current_depth == 0:
        return calibrated_radius  # Fallback if invalid depth
    return int(calibrated_radius * (calibrated_depth / current_depth))

def fit_circle_least_squares(points):
    # Arrange points in the form needed for least squares
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[2*x, 2*y, np.ones(points.shape[0])]
    b = x**2 + y**2
    # Solve least squares problem
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = c
    radius = np.sqrt(c0 + cx**2 + cy**2)
    return (cx, cy, radius)

# ------ Main Loop ------
ball_centers = []
try:
	use_hsv_sliders = False  # Set to True once calibrated

	while True:
		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)

		color_frame = aligned_frames.get_color_frame()
		depth_frame = aligned_frames.get_depth_frame()

		if not color_frame or not depth_frame:
			continue

		color_image = np.asanyarray(color_frame.get_data())
		depth_image = np.asanyarray(depth_frame.get_data())

		display = color_image.copy()
		hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

		h, w, _ = color_image.shape
		centre = (w // 2, h // 2)

		ball_radii = []
		ball_depths = []

		# Then in your main loop:
		if not calibrated:
			display = apply_focus_mask(display, centre, INIT_SEARCH_CIRCLE)

			ball_detected, ball_center, ball_radius = detect_circle_in_centre(color_image)

			if ball_detected:
				if calibration_start_time is None:
					calibration_start_time = time.time()
					# Reset collection arrays when starting a new calibration
					ball_radii = []
					ball_depths = []

				# Draw the detected circle
				if ball_center and ball_radius:
					cv2.circle(display, ball_center, ball_radius, (0, 255, 0), 2)

				# Collect radius and depth samples
				if ball_radius:
					ball_radii.append(ball_radius)
					
					# Get depth at the ball's location
					depth_image = np.asanyarray(depth_frame.get_data())
					x, y = ball_center
					
					# Create a small mask around the center to get average depth
					mask = np.zeros(depth_image.shape, dtype=np.uint8)
					cv2.circle(mask, ball_center, ball_radius, 255, -1)
					
					# Get valid depth values within the circle
					depth_values = depth_image[mask == 255]
					valid_depths = depth_values[depth_values > 0]
					
					if len(valid_depths) > 0:
						current_depth = np.median(valid_depths) * depth_scale * 1000  # in mm
						ball_depths.append(current_depth)

				elapsed = time.time() - calibration_start_time
				progress = min(elapsed / STABILITY_TIME_REQUIRED, 1.0)

				draw_progress_arc(display, centre, INIT_SEARCH_CIRCLE + 10, progress)

				if elapsed >= STABILITY_TIME_REQUIRED:
					# Calculate average radius and depth
					if ball_radii and ball_depths:
						avg_radius = int(np.mean(ball_radii))
						avg_depth = np.mean(ball_depths)
						
						# Store these calibration values for later use
						calibrated_radius = avg_radius
						calibrated_depth = avg_depth
						
						print(f"Calibrated ball radius: {calibrated_radius} pixels at {calibrated_depth:.2f} mm")
					
					ball_hsv, ball_depth = extract_ball_colour_and_depth(color_image, depth_frame, centre)
					print("ball_hsv: ", ball_hsv)
					calibrated = True
					hsv_range = [[max(0, int(ball_hsv[0] - COLOUR_TOLERANCE)), min(179, int(ball_hsv[0] + COLOUR_TOLERANCE))],
				  				[max(0, int(ball_hsv[1] - COLOUR_TOLERANCE)), 255],
				  				[max(0, int(ball_hsv[2] - COLOUR_TOLERANCE)), 255]]

					print(f"Calibrated: HSV={ball_hsv}, Depth={ball_depth:.2f} mm")
			else:
				calibration_start_time = None

		else:
			detected_ball = False

			if use_hsv_sliders:
				h_range, s_range, v_range = colour_slider.get_hsv_from_trackbars()
				hsv_visualization = colour_slider.render_hsv_color_visualization(h_range, s_range, v_range)
				cv2.imshow('HSV Adjustment', hsv_visualization)

				hsv_range = [h_range, s_range, v_range]

			else:
				h_range, s_range, v_range = hsv_range
			
			mask = find_ball_local(depth_image, h_range, s_range, v_range, ball_depth, detected_ball, ball_centers)
			expected_radius = predict_radius(calibrated_radius, calibrated_depth, ball_depth)

			# Method 1: Contour and Min Enclosing Circle
			contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			if contours:
				candidates = []
				for contour in contours:
					cv2.drawContours(display, [contour], -1, (255, 0, 0), 2)

					(x, y), radius = cv2.minEnclosingCircle(contour)
					if radius > 5:  # Filter out noise
						candidates.append((int(x), int(y), int(radius)))

				if candidates:
					best_candidate = None
					smallest_error = float('inf')

					for (cx, cy, radius) in candidates:

						# Get local depth
						if 0 <= cx < w and 0 <= cy < h:
							local_z = depth_image[cy, cx]
							if local_z > 0:
								local_depth = local_z * depth_scale * 1000  # mm
								expected_radius = predict_radius(calibrated_radius, calibrated_depth, local_depth)

								error = abs(radius - expected_radius)

								if error < smallest_error and error < 30:  # 30 pixel tolerance
									smallest_error = error
									best_candidate = (cx, cy, expected_radius, local_depth)

					if best_candidate:
						cx, cy, expected_radius, ball_depth = best_candidate
						detected_ball = True
						depth_detect = False
						cv2.circle(display, (cx, cy), int(expected_radius), (0, 255, 0), 2)
						cv2.putText(display, "Ball detected", (cx - 30, cy - int(expected_radius) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

						ball_centers.append([cx, cy])
						if len(ball_centers) > 2:
							ball_centers.pop(0)

						# Update ball_depth by averaging local region depth
						region = 10  # 10 pixel radius
						local_depths = []
						for dy in range(-region, region + 1):
							for dx in range(-region, region + 1):
								x = cx + dx
								y = cy + dy
								if 0 <= x < w and 0 <= y < h:
									z = depth_image[y, x]
									if z > 0:
										local_depths.append(z * depth_scale * 1000)  # in mm
						if local_depths:
							ball_depth = np.median(local_depths)  # Update depth using median


						if is_ball_in_air(cx, cy, ball_depth, expected_radius, depth_frame):
							# Overlay green if in flight
							overlay = np.full(display.shape, (0, 255, 0), dtype=np.uint8)
							alpha = 0.3  # Transparency
							cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
					

			if not detected_ball:
				cv2.putText(display, "No ball detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

			# --- Visualisation ---
			# Full depth map coloured
			depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

			# Draw the mask region in white
			white_region = np.zeros_like(depth_colormap)
			white_region[:, :] = (255, 255, 255)  # White color
			mask_3c = cv2.merge([mask, mask, mask])
			combined_display = np.where(mask_3c == 255, white_region, depth_colormap)

			# Show focused RGB for reference
			focused_rgb = apply_focus_mask(color_image, ball_centers[-1] if len(ball_centers)>0 else centre, SEARCH_RADIUS)
			cv2.imshow('Focused RGB', focused_rgb)

			# cv2.imshow('Depth Map with Interest Region', combined_display)

		# Always show Ball Tracker window
		cv2.imshow('Ball Tracker', display)

		key = cv2.waitKey(1) & 0xFF

		if key == ord('c'):
			use_hsv_sliders = not use_hsv_sliders
			if use_hsv_sliders:
				colour_slider.create_hsv_trackbars()
				colour_slider.update_hsv_trackbars(hsv_range)
			else:
				cv2.destroyWindow('HSV Adjustment')

		if key == ord('r'):
			calibrated = False
			calibration_start_time = None
			ball_hsv = None
			ball_depth = None
			last_ball_pos = None
			flight_tracking = False
			flight_path = []
			print("Calibration reset.")

		if key == ord('q'):
			break

finally:
	pipeline.stop()
	cv2.destroyAllWindows()