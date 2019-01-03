# WutongFace---A toolkit for face detector and feature extractor using pytorch

## Usage
1. clone this repo and use like:
```python
from wutong_face import WutongFace

wface = WutongFace()
image_file = "F:\\images\\000002.jpg"
### faces_no_align is a list of PIL.Image
faces_no_align = wface.face_no_align(image_file,crop_size=(180,180),min_face_size=20)
#faces_align = wface.face_align(image_file,crop_size=(180,180),min_face_size=20)
faces_no_align[0].show()
```

2. you can also use the base detector api like:
```python
from wutong_face import WutongFace
image_file = "F:\\images\\000002.jpg"
##init a WutongFace
wface = WutongFace()

# 1. using the detector
image = Image.open(image_file)
face_num,bounding_boxes,landmarks = wface.detect(image)
```

3. the perfomance



![](https://raw.githubusercontent.com/wvinzh/picgo-images/image/20181127141157.png)

4. for face feature extract

   ```python
   # 2. use detector and extract feature
   	faces_align,_,_ = wface.face_align(image_file,crop_size=(150,150))
       print(len(faces_align))
       if len(faces_align) <= 0:
           print("no faces")
           
   	feature = wface.feature_extractor('resnet',faces_align)
       print(feature.shape)
   ```



## Reference

[mtcnn-pytorch](https://github.com/polarisZhao/mtcnn-pytorch)

[sphereface-onedrive](https://1drv.ms/u/s!AseTbxZ7P87UjhLteizhWRjJAaDV)  
[cosface-onedrive](https://1drv.ms/u/s!AseTbxZ7P87Ujg8HHy_6iiuZvIad)  
[insightface-onedrive](https://1drv.ms/u/s!AMeTbxZ7P87UjhE)  
[VGGFace2-resnet-onedrive](https://1drv.ms/u/s!AMeTbxZ7P87UjhA)