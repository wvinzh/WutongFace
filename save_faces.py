from wutong_face import WutongFace
import os
import numpy as np
def save_face():
    txt="F:\\face_utils\\todo_name_2000.txt"
    root_folder = "F:\\TestSet\\normal"
    to_root_folder = "F://todo_names"
    to_root_folder_feature = "F://todo_names_feature"
    wface = WutongFace()
    with open(txt,'r',encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip()
            cate = line.split('/')[0]
            image_name = line.split('/')[1]
            print(image_name)
            image_basename = image_name.split('.')[0]
            image_full_path = os.path.join(root_folder,image_name)
            face_list,_,_ = wface.face_align(image_full_path,crop_size=(150,150),min_face_size=20)
            save_path = os.path.join(to_root_folder,cate)
            save_path_feature = os.path.join(to_root_folder_feature,cate)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(save_path_feature):
                os.makedirs(save_path_feature)
            for i,face in enumerate(face_list):
                save_name = '%s_face_sub_%d.jpg' %(image_basename,i)
                save_name_feature = '%s_face_sub_%d.npy' %(image_basename,i)
                face.save(os.path.join(save_path,save_name))
                feature = wface.feature_extractor('resnet',[face])
                np.save(os.path.join(save_path_feature,save_name_feature),feature[0])
            line = f.readline()


save_face()
