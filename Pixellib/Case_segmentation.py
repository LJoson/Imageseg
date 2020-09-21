import pixellib
from pixellib.instance import instance_segmentation
import cv2
import time
instance_seg = instance_segmentation()
instance_seg.load_model("mask_rcnn_coco.h5")
start1 = time.time()
#segment_image.segmentImage("./Images/sample2.jpg", output_image_name= "image_new1.jpg", show_bboxes= True)
segmask, output = instance_seg.segmentImage("./Images/sample2.jpg", show_bboxes= True)
cv2.imwrite("img1.jpg", output)
end1 = time.time()
time1 = end1-start1
print("Inference Time: ",'%.2f'%time1,'seconds')
cv2.namedWindow("img1.jpg",0)
cv2.imshow("img1.jpg", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(output.shape)