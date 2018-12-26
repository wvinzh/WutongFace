# A face feature extractor using pretrained models  
(Now included: SphereFace, CosFace, InsightFace, VGGFACE2-Resnet50)
## Intro  
This repo is used to extract feature from face images and save the feature to npy numpy.
## Usage

1. download pretraned models into ./model  
    [sphereface-onedrive](https://1drv.ms/u/s!AseTbxZ7P87UjhLteizhWRjJAaDV)  
    [cosface-onedrive](https://1drv.ms/u/s!AseTbxZ7P87Ujg8HHy_6iiuZvIad)  
    [insightface-onedrive](https://1drv.ms/u/s!AMeTbxZ7P87UjhE)  
    [VGGFace2-resnet-onedrive](https://1drv.ms/u/s!AMeTbxZ7P87UjhA)

2. clone this repo and coding like this:
   ```python
    image_list = "../*/img_list.txt"
    image_root_folder = "../*/images"
    feature_folder = "../*/featrues"
    extractor = Extractor('cosface') # ['resnet', 'sphereface','cosface', 'insightface']
    extractor.extract(image_root_folder,image_list,feature_folder)
   ```
## Reference

[sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch)  
[CosFace_pytorch](https://github.com/MuggleWang/CosFace_pytorch)  
[InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)  
[VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch)  