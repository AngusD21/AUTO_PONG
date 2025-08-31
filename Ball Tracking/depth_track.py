import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------ Configurations ------
DEPTH_TOLERANCE = 15  # mm
SEARCH_RADIUS = 200
BALL_DIAMETER_MM = 40
INIT_SEARCH_CIRCLE = 70  # pixels
STABILITY_TIME_REQUIRED = 2.0  # seconds

def detect_circle_in_depth(depth_image, depth_scale):
	"""Detect a circular object in the depth image using contour detection and circularity test"""
	# Create a binary mask for valid depth values
	depth_mm = depth_image * depth_scale * 1000
	valid_mask = np.zeros_like(depth_image, dtype=np.uint8)
	
	# Filter depths that could be in a reasonable range (e.g., 0.5m to 3m)
	min_depth = 500  # mm
	max_depth = 3000  # mm
	valid_mask[(depth_mm > min_depth) & (depth_mm < max_depth)] = 255
	
	# Apply morphological operations to clean up the mask
	kernel = np.ones((5, 5), np.uint8)
	valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel)
	
	# Find contours in the depth mask
	contours, _ = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	best_circle = None
	best_circularity = 0
	min_circularity = 0.7  # Threshold for circularity
	
	for contour in contours:
		# Skip small contours
		if cv2.contourArea(contour) < 50:
			continue
			
		# Find the minimum enclosing circle
		(x, y), radius = cv2.minEnclosingCircle(contour)
		center = (int(x), int(y))
		radius = int(radius)
		
		# Calculate circularity (1.0 for a perfect circle)
		area = cv2.contourArea(contour)
		perimeter = cv2.arcLength(contour, True)
		circularity = 0
		if perimeter > 0:
			circularity = 4 * np.pi * area / (perimeter * perimeter)
		
		# Check if it's circular enough and better than previous candidates
		if circularity > min_circularity and circularity > best_circularity:
			# Get the median depth within this contour
			mask = np.zeros_like(depth_image, dtype=np.uint8)
			cv2.drawContours(mask, [contour], 0, 255, -1)
			depth_values = depth_mm[mask == 255]
			valid_depths = depth_values[depth_values > 0]
			
			if len(valid_depths) > 0:
				median_depth = np.median(valid_depths)
				best_circle = (center, radius, median_depth)
				best_circularity = circularity
	
	return best_circle

def depth_detection(depth_frame, depth_scale, ball_depth, expected_radius, prediction, fast_mode):
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
		if area < 100: 
			continue

		(x, y), radius = cv2.minEnclosingCircle(cnt)
		if radius == 0:
			continue

		# Roundness check: area of contour vs area of perfect circle
		circle_area = np.pi * (radius ** 2)
		roundness = area / circle_area

		if roundness < 0.4:  # too irregular
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
	vel_3D, vel_mg_3D, pred_cent_3D, vel_pixel, vel_mg_pixel, pred_cent_pix = None, None, None, None, None, None

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

def display_depth_image(depth_image, depth_scale):
	
	depth_mm = depth_image * depth_scale * 1000
	# Normalize to 0-255 range for display
	norm_depth = np.zeros_like(depth_mm, dtype=np.uint8)
	cv2.normalize(depth_mm, norm_depth, 0, 255, cv2.NORM_MINMAX)
	depth_colormap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
	return depth_colormap

# ------ Main Loop ------
def main():
	# Setup pipeline
	pipeline = rs.pipeline()
	config = rs.config()
	
	# Configure depth stream at 90 FPS
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
	
	# Start pipeline
	profile = pipeline.start(config)
	
	# Get depth sensor scale
	depth_sensor = profile.get_device().first_depth_sensor()
	depth_scale = depth_sensor.get_depth_scale()
	
	# Get intrinsics
	depth_profile = profile.get_stream(rs.stream.depth)
	intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
	
	# Initialize variables
	ball_depth = 0
	radius = 0
	fast_mode = False
	
	ball_centers_pixel = []
	ball_centers_3D = []
	collecting_points = False
	example_points_3d = []
	depth_detection_results = []
	parabola_fits_3d = []

	detected_ball = False
	undetected_count = 0
	detection_count = 0
	
	try:
		while True:
			frame_start_time = time.time()
			timestamps = {}
			
			# Wait for frames
			frames = pipeline.wait_for_frames()
			depth_frame = frames.get_depth_frame()
			timestamps["frame_acquisition"] = time.time()
			
			if not depth_frame:
				continue
				
			# Get depth data
			depth_image = np.asanyarray(depth_frame.get_data())
			
			# Create display image
			if not fast_mode:
				display = depth_image.copy()
				display_color = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
				display_color = display_color.astype(np.uint8)
			h, w = display.shape
			centre = (w // 2, h // 2)
			
			if not detected_ball:
				# Detection using general contour method
				ball_info = detect_circle_in_depth(depth_image, depth_scale)
				
				if ball_info:
					ball_center, radius, detected_depth = ball_info
					cx, cy = ball_center
					
					# Update depth
					ball_depth = detected_depth
					
					# Pixel data
					ball_centers_pixel.append([cx, cy])
					if len(ball_centers_pixel) > 2:
						ball_centers_pixel.pop(0)
					
					# 3D data
					point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], ball_depth)
					ball_centers_3D.append(point_3d)
					if len(ball_centers_3D) > 2:
						ball_centers_3D.pop(0)
					
					detected_ball = True

					if fast_mode:
						depth_detection_results.append(point_3d)
					
					if collecting_points:
						example_points_3d.append(point_3d)
				else:
					cv2.putText(display_color, "No ball detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
				
				timestamps["general_detection"] = time.time()
			
			
			else:
				# Calculate velocities and predictions
				vel_3D, vel_mg_3D, pred_cent_3D, vel_pixel, vel_mg_pixel, pred_cent_pix = get_velocities(
					ball_centers_3D, ball_centers_pixel, centre)
				
				# Use the focused depth detection function
				centre_info, depth_info = depth_detection(depth_frame, depth_scale, ball_depth, radius, pred_cent_pix, fast_mode)
				
				if centre_info is not None:
					cx, cy = centre_info
					ball_depth = depth_info
					
					# Pixel Data
					ball_centers_pixel.append([cx, cy])
					if len(ball_centers_pixel) > 2:
						ball_centers_pixel.pop(0)
					
					# 3D Data
					point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], ball_depth)
					ball_centers_3D.append(point_3d)
					if len(ball_centers_3D) > 2:
						ball_centers_3D.pop(0)
					
					detected_ball = True
					detection_count += 1
					undetected_count = 0
					
					if fast_mode:
						depth_detection_results.append(point_3d)
						
						if len(depth_detection_results) > 5:
							points_np = np.array(depth_detection_results)
							t = np.arange(len(points_np))
							
							# Fit x(t), y(t), z(t)
							coeffs_x = np.polyfit(t, points_np[:, 0], 2)
							coeffs_y = np.polyfit(t, points_np[:, 1], 2)
							coeffs_z = np.polyfit(t, points_np[:, 2], 2)
							
							parabola_fits_3d.append((coeffs_x, coeffs_y, coeffs_z))
					
					if collecting_points:
						example_points_3d.append(point_3d)
				
				else:
					undetected_count += 1
					if undetected_count > 5 or detection_count < 5:
						if len(depth_detection_results) > 10:
							points_np = np.array(depth_detection_results)
							fig = plt.figure()
							ax = fig.add_subplot(111, projection='3d')
							ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=2, c='r', label='Ball Points')
							
							ax.set_xlabel("X (m)")
							ax.set_ylabel("Y (m)")
							ax.set_zlabel("Z (m)")
							ax.set_title("3D Ball Trajectory with Fitted Parabolas")
							
							if len(parabola_fits_3d) > 0:
								colors = plt.cm.plasma(np.linspace(0, 1, len(parabola_fits_3d)))
								
								for i, (cx, cy, cz) in enumerate(parabola_fits_3d):
									t_vals = np.linspace(0, len(depth_detection_results)-1, 100)
									x_vals = np.polyval(cx, t_vals)
									y_vals = np.polyval(cy, t_vals)
									z_vals = np.polyval(cz, t_vals)
									
									ax.plot(x_vals, y_vals, z_vals, color=colors[i], label=f'Fit {i+1}')
								
								ax.legend()
							
							plt.show()
						
						# Reset tracking
						depth_detection_results = []
						parabola_fits_3d = []
						ball_centers_pixel = []
						ball_centers_3D = []
						detected_ball = False
						undetected_count = 0
						detection_count = 0
				
				# Visualization
				if not fast_mode:
					if detection_count > 5:
						
						cv2.putText(display_color, "Ball detected", (cx - 30, cy - int(radius) - 10), 
									cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
						
						cv2.circle(display_color, centre_info, radius, (0, 0, 255), 2)
						overlay = np.full(display_color.shape, (0, 255, 0), dtype=np.uint8)
						alpha = 0.3  # Transparency
						cv2.addWeighted(overlay, alpha, display_color, 1 - alpha, 0, display_color)
				
				timestamps["depth_detection"] = time.time()
			
			# Display frame in slow mode
			if not fast_mode:
				if collecting_points:
					cv2.putText(display_color, "Collecting", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
				cv2.imshow('Depth Ball Tracker', display_color)
			
			# Process keyboard input
			key = cv2.waitKey(1) & 0xFF
			
			if key == ord('r'):
				ball_depth = None
				detected_ball = False
				print("Calibration reset.")
			
			if key == ord('h'):
				collecting_points = not collecting_points
				if not collecting_points and len(example_points_3d) > 0:
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
					cv2.putText(display_color, "Fast Mode", (h//2, w//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
					display_color = apply_focus_mask(display_color, centre, 1)
					cv2.imshow('Depth Ball Tracker', display_color)
					cv2.waitKey(100)
					print("Fast mode enabled.")
				else:
					print("Slow mode enabled.")
			
			if key == ord('q'):
				break
			
			# Print frame performance info
			if not fast_mode:
				frame_end_time = time.time()
				timestamps["end_of_frame"] = frame_end_time
				
				print("\n--- Frame Timing ---")
				prev_time = frame_start_time
				for label, ts in timestamps.items():
					print(f"{label:<20}: {(ts - prev_time)*1000:.2f} ms")
					prev_time = ts
				print(f"{'TOTAL':<20}: {(frame_end_time - frame_start_time)*1000:.2f} ms")
	
	finally:
		pipeline.stop()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	main()