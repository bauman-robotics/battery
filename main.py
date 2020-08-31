#Import lib
import pyrealsense2 as rs
import cv2
import numpy as np
import jetson.utils
import jetson.inference

#Configure object detection net
net = jetson.inference.detectNet(argv=['--model=ssd-mobilenet.onnx',
					'--labels=labels.txt',
					'--input-blob=input_0',
					'--output-cvg=scores',
					'--output-bbox=boxes',
					'--alpha=0',
					'--threshold=0.5'])

#Configure depth and collor streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
colorizer = rs.colorizer()

#Start streaming
profile = pipeline.start(config)

#Gettting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

#Create a aligh object
align =  rs.align(rs.stream.color)

#Streaming loop
try:
	while True:
		#Align color and depth frame
		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)
		aligned_color_frame = aligned_frames.get_color_frame()
		aligned_depth_frame = aligned_frames.get_depth_frame()
		if not aligned_color_frame or not aligned_depth_frame:
			continue
		#Convert to numpy format
		color_image = np.asanyarray(aligned_color_frame.get_data())
		depth_image = np.asanyarray(aligned_depth_frame.get_data())

		#Detect with numpy->CUDA->numpy convertion
		rgb_img = jetson.utils.cudaFromNumpy(color_image)
		detections = net.Detect(rgb_img, overlay='none')
		bgr_img = jetson.utils.cudaAllocMapped(width=rgb_img.width,
							height=rgb_img.height,
							format='bgr8')
		jetson.utils.cudaConvertColor(rgb_img, bgr_img)
		jetson.utils.cudaDeviceSynchronize()
		cv_color_img = jetson.utils.cudaToNumpy(bgr_img)

		#Produsing depth colormap with rslib func
		depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

		for det in detections:
			#mean depth value into the box
			depth = depth_image[int(det.Left):int(det.Right), int(det.Top):int(det.Bottom)]
			depth = depth *  depth_scale
			dist, _, _, _ = cv2.mean(depth)

			#draw info about detection
			cv2.rectangle(cv_color_img, (int(det.Left), int(det.Top)),
							(int(det.Right), int(det.Bottom)),
							(255, 255, 255), 2)
			cv2.putText(cv_color_img, "{1:.3} meters away".format(det.ClassID, dist),
							(int(det.Left), int(det.Top) - 5),
							cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255))
			cv2.rectangle(depth_colormap, (int(det.Left), int(det.Top)),
							(int(det.Right), int(det.Bottom)),
							(255, 255, 255), 2)
		#Output wwindow formation
		images = np.hstack((cv_color_img, depth_colormap))

		#Interrupt
		cv2.imshow('Aligh examples', images)
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
finally:
	pipeline.stop()



