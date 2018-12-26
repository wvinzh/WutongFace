import os
from PIL import Image
import numpy as np
from face_detector import detect_faces
from face_detector.align_trans import get_reference_facial_points, warp_and_crop_face
import cv2
from feature_extractor import extractor
import torch
from sklearn import svm
import pickle
from classifier.classifier_net import Classifier
import torch.nn.functional as F
def read_npy(npy_path):
    return np.load(npy_path).reshape(-1)

def load_dataset(file_root, train_txt, val_txt):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    with open(train_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        file_path, label = line.strip().split()
        train_x.append(read_npy(os.path.join(file_root, file_path)))
        train_y.append(int(label))
    with open(val_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        file_path, label = line.strip().split()
        val_x.append(read_npy(os.path.join(file_root, file_path)))
        val_y.append(int(label))
    res = {}
    with open(train_txt,'r',encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            name = line.split()[0].split('/')[0]
            label = int(line.split()[1])
            if label not in res and label != -1:
                res[label] = name
                # print(name)
            line = f.readline()
    return np.array(train_x), np.array(train_y), np.array(val_x), np.array(val_y),res



class WutongFace(object):

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def detect(self, image, min_face_size=20.0,
               thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7]):
        bounding_boxes, landmarks = detect_faces(image, min_face_size=min_face_size, thresholds=thresholds,nms_thresholds=nms_thresholds)
        # print('aaaa',bounding_boxes)
        return len(bounding_boxes), bounding_boxes, landmarks

    def face_align(self, img_path, crop_size=(150, 150), min_face_size=20.0,
                   thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7]):
        assert os.path.exists(img_path), "No Such File %s" % img_path
        image = Image.open(img_path).convert('RGB')
        face_nums, bounding_boxes, landmarks = self.detect(image,min_face_size=min_face_size,thresholds=thresholds,nms_thresholds=nms_thresholds)
        # print(face_nums)
        res_faces = []
        if face_nums <= 0:
            return res_faces,bounding_boxes, landmarks
        refrence = get_reference_facial_points(inner_padding_factor=1e-8,default_square=True,output_size=crop_size)
        for i in range(len(bounding_boxes)):
            landmark = landmarks[i]
            facial5points = [[landmark[j], landmark[j+5]] for j in range(5)]
            warped_face = warp_and_crop_face(
                np.array(image), facial5points, refrence, crop_size=crop_size)
            warped_face = Image.fromarray(warped_face)
            res_faces.append(warped_face)
        return res_faces,bounding_boxes, landmarks

    def face_no_align(self, img_path, crop_size=(150, 150), min_face_size=20.0,
                   thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7]):
        assert os.path.exists(img_path), "No Such File %s" % img_path
        image = Image.open(img_path)
        face_nums, bounding_boxes, landmarks = self.detect(image,min_face_size=min_face_size,thresholds=thresholds,nms_thresholds=nms_thresholds)
        image_np = np.array(image)
        # image_np = cv2
        res_faces = []
        # print(image_np.shape)
        for bb in bounding_boxes:
            bb = self.get_size_diff(crop_size,bb) ## 0,1,2,3 left, bottom, right, top
            box = [int(b) for b in bb[0:4]]
            faces = image_np[box[1]:box[3],box[0]:box[2]]
            # print(faces.shape)
            pil_face = Image.fromarray(faces)
            pil_face=pil_face.resize(crop_size, Image.ANTIALIAS)
            # pil_face.show()
            res_faces.append(pil_face)
        # for i in range(len)
        return res_faces,bounding_boxes, landmarks
    
    def get_size_diff(self,crop_size,bounding_box):
        h = bounding_box[3] - bounding_box[1]
        w = bounding_box[2] - bounding_box[0]
        rate_t = float(crop_size[1] / crop_size[0]) #target rate
        rate_o = float(h / w) #original rate
        print(rate_o,rate_t)
        if rate_t <= rate_o:
            h_t = h
            w_t = h/rate_t
        elif rate_t > rate_o:
            h_t = w*rate_t
            w_t = w  
            print(w_t,h_t) 
        w_diff = w_t - w
        h_diff = h_t - h
        # print(w_diff,h_diff)
        new_bounding_box = [bounding_box[0]-w_diff/2,bounding_box[1]-h_diff/2,bounding_box[2]+w_diff/2,bounding_box[3]+h_diff/2]
        # print(bounding_box,new_bounding_box)
        return new_bounding_box

    def feature_extractor(self, model_name, pil_face_image_list):
        et = extractor.Extractor(model_name)
        model  = et.model.eval()
        transform  = et.transform
        input = [transform(pil_face_image) for pil_face_image in pil_face_image_list]
        # input = input.unsqueeze(0) # C,H,W -> B,C,H,W
        input = torch.stack(input,0)
        output = model(input)
        feature = output.data.cpu().numpy()
        feature = feature.reshape(feature.shape[0],feature.shape[1])
        return feature

    def train_classifier(self):
        ##train svm classifier
        file_root = "F:\\final_faces_feature_resnet"
        train_txt = "F:\\images\\train_with_normal.txt"
        val_txt = "F:\\images\\val_with_normal.txt"
        train_x, train_y, val_x, val_y, name_dict = load_dataset(
            file_root, train_txt, val_txt)
        svm_c = svm.SVC(kernel="linear")
        svm_c.fit(train_x, train_y)
        # print(name_dict)
        with open('classifier_svm.pkl','wb') as f:
            pickle.dump(svm_c,f)
        with open('annotatioan.pkl','wb') as f:
            pickle.dump(name_dict,f)
        # return svm_c,name_dict

    def get_classifier(self):
        with open('classifier_svm.pkl','rb') as f:
            svm_c = pickle.load(f)
        with open('annotatioan.pkl','rb') as f:
            name_dict=pickle.load(f)
        return svm_c,name_dict
    
    def NN_classifier(self):
        '''Neural Network classifier
        '''

        c = Classifier(in_features=2048,num_class=145)

        c.load_state_dict(torch.load("classifier-best.pth"))
        # with open("classifier-best-512-0.9768.pth",'rb') as f:
        #     torch.load(f)
        # print('ok')
        with open('annotatioan.pkl','rb') as f:
            name_dict=pickle.load(f)
        def clsf(x):
            x = torch.from_numpy(x)
            out = c(x)
            pred = F.softmax(out,dim=1)
            pred = torch.max(pred,dim=1)[1]
            pred = pred - 1
            return pred.numpy()
        return clsf,name_dict

if __name__ == '__main__':
    image_file = "F:\\PoliticsTestSet\\cover\\5705714_rs16_4.png"
    # image_file = "F:\\PoliticsTestSet\\cover/1136006_54fe2ec1ecec202f48a379f1950ad8f6_big.jpg"
    ##init a WutongFace
    wface = WutongFace()

    # 1. using the detector
    # image = Image.open(image_file)
    # image.show()
    # face_num,bounding_boxes,landmarks = wface.detect(image)

    # 2. use detector and crop with align
    faces_align,_,_ = wface.face_align(image_file,crop_size=(150,150))
    # print(len(faces_align))
    if len(faces_align) <= 0:
        print("no faces")
        
    feature = wface.feature_extractor('resnet',faces_align)
    # print(feature.shape)
    # 3. use detector and crop with no align
    # faces_no_align = wface.face_no_align(image_file,crop_size=(180,180))
    # faces_no_align[0].show()
    #face_iamge_list is a list of PIL.Image file

    #4. train classifier(just need the first time)
    # wface.train_classifier()
    svm_c, name_dict = wface.NN_classifier()
    #5. predict
    pred = svm_c(feature)
    print(pred)
    for p in pred:
        if p in name_dict:
            print(name_dict[p])
