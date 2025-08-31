import time
from matplotlib import pyplot as plt
import pyrealsense2 as rs
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import colour_slider
from kalman_filter import KalmanFilter3D

# ------ Configurations ------
COLOUR_TOLERANCE = 10  # HSV threshold range
DEPTH_TOLERANCE = 15  
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
color_profile = profile.get_stream(rs.stream.color)
intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

# Align depth to colour
align = rs.align(rs.stream.color)

# Get depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

kf = KalmanFilter3D()

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
def find_ball_local(depth_img, h_range, s_range, v_range, ball_depth, ball_detected, predicted_cent):
	
	# Colour mask
	lower = np.array([h_range[0], s_range[0], v_range[0]])
	upper = np.array([h_range[1], s_range[1], v_range[1]])
	mask = cv2.inRange(hsv_image, lower, upper)

	# Depth threshold
	depth_in_mm = depth_img * depth_scale * 1000
	depth_mask = np.logical_and(depth_in_mm > (ball_depth - DEPTH_TOLERANCE),
								depth_in_mm < (ball_depth + DEPTH_TOLERANCE))
	depth_mask = depth_mask.astype(np.uint8) * 255

	# Prediction Area
	search_mask = np.zeros_like(mask)
	cx, cy = predicted_cent
	cv2.circle(search_mask, (cx, cy), SEARCH_RADIUS, 255, -1)

	if not ball_detected:
		combined_mask = mask
	else:
		combined_mask = cv2.bitwise_and(mask, depth_mask)
		combined_mask = cv2.bitwise_and(combined_mask, search_mask)

	return combined_mask


# def depth_detection(depth_frame, ball_depth, last_centres):
# 	depth_img = np.asanyarray(depth_frame.get_data())
# 	h, w = depth_image.shape

# 	# Depth threshold
# 	depth_in_mm = depth_img * depth_scale * 1000
# 	depth_mask = np.logical_and(depth_in_mm > (ball_depth - DEPTH_TOLERANCE),
# 								depth_in_mm < (ball_depth + DEPTH_TOLERANCE))
# 	depth_mask = depth_mask.astype(np.uint8) * 255

# 	if len(last_centres) > 0:
# 		# Limit search area around last known centre
# 		search_mask = np.zeros_like(depth_mask)
# 		cx, cy = last_centres[-1]

# 		cv2.circle(search_mask, (cx, cy), SEARCH_RADIUS, 255, -1)
		
# 		if len(last_centres) > 1:
# 			vel = np.array(last_centres[-1]) - np.array(last_centres[-2])

# 			predicted_cent = last_centres[-1] + vel
# 			cx2, cy2 = predicted_cent
# 			cv2.circle(search_mask, (cx2, cy2), SEARCH_RADIUS, 255, -1)
		
# 		depth_mask = cv2.bitwise_and(depth_mask, search_mask)
	
# 	# Search within mask for circular shape
# 	contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def depth_detection(depth_frame, ball_depth, expected_radius, prediction, fast_mode):
	depth_img = np.asanyarray(depth_frame.get_data())

	# Depth threshold
	depth_in_mm = depth_img * depth_scale * 1000
	depth_mask = np.logical_and(
		depth_in_mm > (ball_depth - DEPTH_TOLERANCE),
		depth_in_mm < (ball_depth + DEPTH_TOLERANCE)
	).astype(np.uint8) * 255

	# Prediction mask
	search_mask = np.zeros_like(depth_mask)
	cv2.circle(search_mask, (int(prediction[0]), int(prediction[1])), SEARCH_RADIUS, 255, -1)
			
	depth_mask = cv2.bitwise_and(depth_mask, search_mask)
	
	if not fast_mode:
		cv2.imshow("Depth Mask", depth_mask)
		cv2.imshow("Search Mask", search_mask)

	# Search within mask for contours
	contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	best_contour = None
	best_score = float('inf')
	found_center = None
	found_avg_depth = None

	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area < 50:  # discard tiny noise blobs
			continue

		(x, y), radius = cv2.minEnclosingCircle(cnt)
		if radius == 0:
			continue

		# Roundness check: area of contour vs area of perfect circle
		circle_area = np.pi * (radius ** 2)
		roundness = area / circle_area

		if roundness < 0.6:  # too irregular
			continue

		radius_diff = abs(radius - expected_radius)

		# Combined score: prioritize radius closeness
		if radius_diff < 10 and radius_diff < best_score:
			best_score = radius_diff
			best_contour = cnt
			found_center = (int(x), int(y))

	if best_contour is not None and found_center is not None:
		# Create a mask for the detected ball
		ball_mask = np.zeros_like(depth_mask)
		cv2.drawContours(ball_mask, [best_contour], -1, 255, -1)

		# Extract average depth inside the ball region
		ball_pixels = depth_in_mm[ball_mask == 255]

		ball_pixels = ball_pixels[np.logical_and(
			ball_pixels > (ball_depth - DEPTH_TOLERANCE),
			ball_pixels < (ball_depth + DEPTH_TOLERANCE)
		)]

		if len(ball_pixels) > 0:
			found_avg_depth = np.median(ball_pixels)

		if not fast_mode:
			cv2.circle(depth_mask, found_center, int(expected_radius), (0, 255, 0), 2)
			cv2.imshow("Detected Ball", depth_mask)

		return found_center, found_avg_depth

	# If no ball found
	return None, None


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


def get_velocities(ball_centers_3D, ball_centers_pixel, centre):

	vel_3D, vel_mg_3D, pred_cent_3D, vel_pixel, vel_mg_pixel, pred_cent_pix =\
	None, None, None, None, None, None

	# 3D Vel
	if len(ball_centers_3D) > 1:
		vel_3D = np.array(ball_centers_3D[-1]) - np.array(ball_centers_3D[-2])
		vel_mg_3D = np.linalg.norm(vel_3D)

		pred_cent_3D = ball_centers_3D[-1] + vel_3D

	# Pixel Vel
	if len(ball_centers_pixel) > 1:
		vel_pixel = np.array(ball_centers_pixel[-1]) - np.array(ball_centers_pixel[-2])
		vel_mg_pixel = np.linalg.norm(vel_pixel)

		pred_cent_pix = ball_centers_pixel[-1] + vel_pixel
	else:
		pred_cent_pix = ball_centers_pixel[-1] if len(ball_centers_pixel) > 0 else centre
	

	return vel_3D, vel_mg_3D, pred_cent_3D, vel_pixel, vel_mg_pixel, pred_cent_pix



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
ball_centers_pixel = []
ball_centers_3D = []
try:
	use_hsv_sliders = False  # Set to True once calibrated
	rgb_detect = True
	depth_detect = False
	fast_mode = False

	collecting_points = False
	example_points_3d = []
	depth_detection_results = []
	parabola_fits_3d = []

	while True:
		frame_start_time = time.time()
		timestamps = {}

		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)

		if rgb_detect:
			color_frame = aligned_frames.get_color_frame()
		depth_frame = aligned_frames.get_depth_frame()

		timestamps["frame_acquisition"] = time.time()

		if not color_frame or not depth_frame:
			continue

		color_image = np.asanyarray(color_frame.get_data())
		depth_image = np.asanyarray(depth_frame.get_data())

		display = color_image.copy()
		hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

		timestamps["hsv_conversion"] = time.time()

		h, w, _ = color_image.shape
		centre = (w // 2, h // 2)

		detected_ball = False
		ball_radii = []
		ball_depths = []

		# Calibration
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
			
			timestamps["calibration"] = time.time()
		
		# Colour Detection
		elif rgb_detect:

			if use_hsv_sliders:
				h_range, s_range, v_range = colour_slider.get_hsv_from_trackbars()
				hsv_visualization = colour_slider.render_hsv_color_visualization(h_range, s_range, v_range)
				cv2.imshow('HSV Adjustment', hsv_visualization)

				hsv_range = [h_range, s_range, v_range]

			else:
				h_range, s_range, v_range = hsv_range

			# Expected radius based on previous depth
			expected_radius = predict_radius(calibrated_radius, calibrated_depth, ball_depth)

			# Velocity and predicted position
			vel_3D, vel_mg_3D, pred_cent_3D, vel_pixel, vel_mg_pixel, pred_cent_pix = get_velocities(ball_centers_3D, ball_centers_pixel, centre)
			
			# Ball Mask
			mask = find_ball_local(depth_image, h_range, s_range, v_range, ball_depth, detected_ball, pred_cent_pix)
			

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

						# Draw detection
						cv2.circle(display, (cx, cy), int(expected_radius), (0, 255, 0), 2)
						cv2.putText(display, "Ball detected", (cx - 30, cy - int(expected_radius) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

						# Depth Data
						region = 5 
						x_min = max(cx - region, 0)
						x_max = min(cx + region + 1, w)
						y_min = max(cy - region, 0)
						y_max = min(cy + region + 1, h)

						local_region = depth_image[y_min:y_max, x_min:x_max]
						valid_depths = local_region[local_region > 0]
						ball_depth = np.median(valid_depths) * depth_scale * 1000

						# Pixel Data
						ball_centers_pixel.append([cx, cy])
						if len(ball_centers_pixel) > 2:
							ball_centers_pixel.pop(0)
						
						# 3D Data
						point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], ball_depth)

						# predicted_position = kf.predict()
						# kf.update(point_3d)
						# smoothed_position = kf.x[:3].flatten()
						# ball_centers_3D.append(smoothed_position.tolist())

						ball_centers_3D.append(point_3d)
						if len(ball_centers_3D) > 2:
							ball_centers_3D.pop(0)

						if collecting_points:
							example_points_3d.append(point_3d)

						# Show focused RGB for reference
						# focused_rgb = apply_focus_mask(color_image, ball_centers[-1] if len(ball_centers)>0 else centre, SEARCH_RADIUS)
						# cv2.imshow('Focused RGB', focused_rgb)

			if vel_mg_3D is not None:
				if vel_mg_3D > 8:
					depth_detect = True
			
			timestamps["rgb_detection"] = time.time()
							
		if not detected_ball and rgb_detect and calibrated:
			cv2.putText(display, "No ball detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

		if depth_detect:
			
			# Velocities and predictions
			vel_3D, vel_mg_3D, pred_cent_3D, vel_pixel, vel_mg_pixel, pred_cent_pix = get_velocities(ball_centers_3D, ball_centers_pixel, centre)
				
			centre, depth = depth_detection(depth_frame, ball_depth, expected_radius, pred_cent_pix, fast_mode)

			if centre is not None:
				cx, cy = centre

				ball_depth = depth

				# Pixel Data
				ball_centers_pixel.append([cx, cy])
				if len(ball_centers_pixel) > 2:
					ball_centers_pixel.pop(0)
				
				# 3D Data
				point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], ball_depth)

				# predicted_position = kf.predict()
				# kf.update(point_3d)
				# smoothed_position = kf.x[:3].flatten()
				# ball_centers_3D.append(smoothed_position.tolist())

				ball_centers_3D.append(point_3d)
				if len(ball_centers_3D) > 2:
					ball_centers_3D.pop(0)
				
				# Visualisation
				if not fast_mode:
					cv2.circle(display, centre, expected_radius, (0, 0, 255), 2)
					overlay = np.full(display.shape, (0, 255, 0), dtype=np.uint8)
					alpha = 0.3  # Transparency
					cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)

				rgb_detect = False

				if fast_mode:
					depth_detection_results.append(point_3d)

					if len(depth_detection_results) > 5:
						points_np = np.array(depth_detection_results)
						t = np.arange(len(points_np))  # treat each point as occurring at step t

						# Fit x(t), y(t), z(t)
						coeffs_x = np.polyfit(t, points_np[:, 0], 2)
						coeffs_y = np.polyfit(t, points_np[:, 1], 2)
						coeffs_z = np.polyfit(t, points_np[:, 2], 2)

						parabola_fits_3d.append((coeffs_x, coeffs_y, coeffs_z))

				if collecting_points:
					example_points_3d.append(point_3d)

				# Show focused RGB for reference
				# focused_rgb = apply_focus_mask(color_image, ball_centers[-1] if len(ball_centers)>0 else centre, SEARCH_RADIUS)
				# cv2.imshow('Focused RGB', focused_rgb)
			
			else:
				print(len(depth_detection_results))
				if len(depth_detection_results) > 10:
					points_np = np.array(depth_detection_results)
					fig = plt.figure()
					ax = fig.add_subplot(111, projection='3d')
					ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=2, c='r', label='Ball Points')

					ax.set_xlabel("X (m)")
					ax.set_ylabel("Y (m)")
					ax.set_zlabel("Z (m)")
					ax.set_title("3D Ball Trajectory with Fitted Parabolas")

					if 'parabola_fits_3d' in globals():
						colors = plt.cm.plasma(np.linspace(0, 1, len(parabola_fits_3d)))

						for i, (cx, cy, cz) in enumerate(parabola_fits_3d):
							t_vals = np.linspace(0, len(depth_detection_results)-1, 100)
							x_vals = np.polyval(cx, t_vals)
							y_vals = np.polyval(cy, t_vals)
							z_vals = np.polyval(cz, t_vals)

							ax.plot(x_vals, y_vals, z_vals, color=colors[i], label=f'Fit {i+1}')

						ax.legend()

					plt.show()

					# Reset
				depth_detection_results = []
				parabola_fits_3d = []

				ball_centers = []
				depth_detect = False
				rgb_detect = True

		timestamps["depth_detection"] = time.time()


		# --- Visualisation ---
		# Full depth map coloured
		# depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

		# # Draw the mask region in white
		# white_region = np.zeros_like(depth_colormap)
		# white_region[:, :] = (255, 255, 255)  # White color
		# mask_3c = cv2.merge([mask, mask, mask])
		# combined_display = np.where(mask_3c == 255, white_region, depth_colormap)
		# cv2.imshow('Depth Map with Interest Region', combined_display)

		
		# Main ball tracker visualisation
		if not fast_mode:
			if collecting_points:
				cv2.putText(display, "Collecting", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
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

		if key == ord('h'):
			collecting_points = not collecting_points
			if not collecting_points:
				points_np = np.array(example_points_3d)
				if points_np.shape[0] > 0:
					fig = plt.figure()
					ax = fig.add_subplot(111, projection='3d')
					ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=2, c='r')
					ax.set_xlabel("X (m)")
					ax.set_ylabel("Y (m)")
					ax.set_zlabel("Z (m)")
					ax.set_title("Collected 3D Ball Points")
					plt.show()

				example_points_3d = []
		
		if key == ord('f'):
			fast_mode = not fast_mode
			if fast_mode:
				cv2.putText(display, "Fast Mode", (h//2, w//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
				display = apply_focus_mask(display, centre, 1)
				cv2.imshow('Ball Tracker', display)
				cv2.waitKey(100)
				print("Fast mode enabled.")


		if key == ord('q'):
			break
	
		# if not fast_mode:
		frame_end_time = time.time()
		timestamps["end_of_frame"] = frame_end_time

		# Print frame performance info
		if not fast_mode:
			print("\n--- Frame Timing ---")
			prev_time = frame_start_time
			for label, ts in timestamps.items():
				print(f"{label:<20}: {(ts - prev_time)*1000:.2f} ms")
				prev_time = ts
			print(f"{'TOTAL':<20}: {(frame_end_time - frame_start_time)*1000:.2f} ms")

finally:
	pipeline.stop()
	cv2.destroyAllWindows()