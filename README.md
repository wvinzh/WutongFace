# WutongFace---A pytorch face detector based on mtcnn

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


## Reference

[mtcnn-pytorch](https://github.com/polarisZhao/mtcnn-pytorch)
