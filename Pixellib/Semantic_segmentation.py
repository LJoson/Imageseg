import pixellib
from pixellib.semantic import semantic_segmentation
import time
import cv2
segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')
start = time.time()
#segment_image.segmentAsPascalvoc('./Images/resize2.jpg', output_image_name= 'image_new.jpg',overlay= True)
segmap, segoverlay = segment_image.segmentAsPascalvoc("./Images/resize2.jpg", overlay= True)
cv2.imwrite("img.jpg", segoverlay)
end = time.time()
time = end-start
print("Inference Time: ",'%.2f'%time,'seconds')
cv2.namedWindow("img.jpg",0)
cv2.imshow("img.jpg", segoverlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(segoverlay.shape)

