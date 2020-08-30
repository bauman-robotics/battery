#Import lib
import pyrealsense2 as rs
import cv2
import numpy as np

#Configure object detection net

#Configure depth and collor streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

#Start streaming
profile = pipeline.start(config)

#Gettting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

#We will clip object more than clipping_distance_in_meters meters away
#clipping_distance_in_meters = 1
#clipping_distance = clipping_distance_in_meters / depth_scale

#Create a aligh object
align =  rs.align(rs.stream.color)

#Streaming loop
try:
	while True:
		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)
		aligned_color_frame = aligned_frames.get_color_frame()
		aligned_depth_frame = aligned_frames.get_depth_frame()
		if not aligned_color_frame or not aligned_depth_frame:
			continue
		color_image = np.asanyarray(aligned_color_frame.get_data())
		depth_image = np.asanyarray(aligned_depth_frame.get_data())

		#grey_color = 150
		#depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
		#bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

		#depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		#images = np.hstack((bg_removed, depth_colormap))

		colorizer =  rs.colorizer()
		depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
		images = np.hstack((color_image, depth_colormap))

		cv2.imshow('Aligh examples', images)
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
finally:
	pipeline.stop()



