import os
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from transformers import AutoModel
from transformers import AutoProcessor
from transformers import TextStreamer

import torch
import torch.nn as nn

from PIL import Image
import requests


# Image Cropper (Using YOLOv8 to detect and crop image objects)
class ImageCropperByYOLOv8(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = YOLO("yolov8n.pt")
        self.names = self.model.names
    def cropping(self, option, image_path=None, video_path=None):
        if (option == 0): # Crop for Image
            im0 = cv2.imread(image_path) 

            crop_dir_name = "Image_crop"
            if not os.path.exists(crop_dir_name):
                os.mkdir(crop_dir_name)

            results = self.model.predict(im0, show=False)

            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names
            annotator = Annotator(im0, line_width=2, example=self.names)

            idx = 0
            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    idx += 1
                    annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])

                    crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

                    cv2.imwrite(os.path.join(crop_dir_name, names[cls] + str(idx) + ".png"), crop_obj)

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

class ImageUnderstanding(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("visheratin/MC-LLaVA-3b", 
                                               torch_dtype=torch.float16, 
                                               trust_remote_code=True).to("cuda")
        self.processor = AutoProcessor.from_pretrained("visheratin/MC-LLaVA-3b", 
                                                       trust_remote_code=True)
    
    def process(self, image_path=None):
        raw_image = Image.open(image_path)

        prompt = """<|im_start|>user
                    <image>
                    Describe the image.<|im_end|>
                    <|im_start|>assistant
                """
        
        with torch.inference_mode():
            inputs = self.processor(prompt, 
                                    [raw_image], 
                                    self.model, 
                                    max_crops=100, 
                                    num_tokens=728)
            
        streamer = TextStreamer(self.processor.tokenizer)
        with torch.inference_mode():
            output = self.model.generate(**inputs, 
                                         max_new_tokens=200, 
                                         do_sample=True, 
                                         use_cache=False, 
                                         top_p=0.9, 
                                         temperature=1.2, 
                                         eos_token_id=self.processor.tokenizer.eos_token_id, 
                                         streamer=streamer)
        return self.processor.tokenizer.decode(output[0]).replace(prompt, "").replace("<|im_end|>", "")