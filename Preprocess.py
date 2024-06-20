import os
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

import torch
import torch.nn as nn


# Image Cropper (Using YOLOv8 to detect and crop image objects)
class ImageCropperByYOLOv8(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = YOLO("yolov8n.pt")
        self.names = self.model.names
    def cropping(self, option, image_path=None, video_path=None):
        if (option == 0): # Crop for Image
            image = cv2.imread() 

            crop_dir_name = "Image_crop"
            if not os.path.exists(crop_dir_name):
                os.mkdir(crop_dir_name)

            results = self.model.predict(im0, show=False)
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            annotator = Annotator(im0, line_width=2, example=self.names)

            idx = 0
            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    idx += 1
                    annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])

                    crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

                    cv2.imwrite(os.path.join(crop_dir_name, cls ,str(idx) + ".png"), crop_obj)

            image.release()
            cv2.destroyAllWindows()

        if (option == 1):  # Crop for Video
            cap = cv2.VideoCapture(video_path)

            w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                                   cv2.CAP_PROP_FRAME_HEIGHT, 
                                                   cv2.CAP_PROP_FPS))
            
            crop_dir_name = "Image_crop"
            if not os.path.exists(crop_dir_name):
                os.mkdir(crop_dir_name)
            
            # Video writer
            video_writer = cv2.VideoWriter("object_cropping_output.avi", 
                                            cv2.VideoWriter_fourcc(*"mp4v"), 
                                            fps, 
                                            (w, h))
            
            idx = 0
            while cap.isOpened():
                success, im0 = cap.read()
                if not success:
                    print("Video frame is empty or video processing has been successfully completed.")
                    break

                results = self.model.predict(im0, show=False)
                boxes = results[0].boxes.xyxy.cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                annotator = Annotator(im0, line_width=2, example=self.names)

                if boxes is not None:
                    for box, cls in zip(boxes, clss):
                        idx += 1
                        annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])

                        crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

                        cv2.imwrite(os.path.join(crop_dir_name, cls ,str(idx) + ".png"), crop_obj)

                # cv2.imshow("ultralytics", im0)
                video_writer.write(im0)

            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()