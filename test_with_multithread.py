from multiprocessing import Process,Queue,Manager
import os
from wutong_face import WutongFace
import time
##主进程负责构建Queue
def get_image_list_Q(q,image_list):
    for image in image_list:
        q.put(image)

def test(wface,q,res):
    while True:
        try:
            image = q.get(block=False)
            print(image)
            faces_align,_,_ = wface.face_align(image,min_face_size=10,crop_size=(150,150))
            res[1] += 1
            if len(faces_align) <= 0: 
                return
            features = wface.feature_extractor('resnet',faces_align)

            classifier,name_dict = wface.get_classifier()

            pred = classifier.predict(features)

            for p in pred:
                if p in name_dict:
                    res[0] += 1
                    print(name_dict[p])
        except Exception as e:
            return

def test2(wface,image_list,res):
    # print("######################### %d" % len(image_list))
    for image in image_list:
        # image = q.get(block=False)
        print(image,res[0],res[1])
        faces_align,_,_ = wface.face_align(image,min_face_size=10,crop_size=(150,150))
        res[1] += 1
        if len(faces_align) <= 0: 
            continue
        features = wface.feature_extractor('resnet',faces_align)

        classifier,name_dict = wface.NN_classifier()

        pred = classifier(features)

        for p in pred:
            if p in name_dict:
                res[0] += 1
                print(name_dict[p])

def main():
    
    q = Queue()
    nnn_txt = "F:\\Test_N.txt"
    ppp_txt = "F:\\Test_P.txt"
    root = "F:\\TestSet"
    process_num = 7
    wface = WutongFace()
    m = Manager()
    res = m.list([0,0]) # res[0] error ; res[1] total
    with open(nnn_txt,'r',encoding='utf-8') as f:
        normals = f.readlines()
        normals = [os.path.join(root,n.strip().rsplit(' ',1)[0]) for n in normals]
    with open(ppp_txt,'r',encoding='utf-8') as f:
        politicals = f.readlines()
        politicals = [os.path.join(root,n.strip().rsplit(' ',1)[0]) for n in politicals]
    image_list_sp = []
    image_list = normals
    # print("================%d"%len(image_list))
    each_pro = int(len(image_list)/process_num)
    for i in range(process_num):
        if i == process_num-1:
            image_list_sp.append(image_list[i*each_pro:])
        else:
            image_list_sp.append(image_list[i*each_pro:(i+1)*each_pro])
    # p1 = Process(target=get_image_list_Q,args=(q,politicals,))
    predict_p = []
    # predict_p.append(p1)
    for i in range(process_num):
        predict_p.append(Process(target=test2,args=(wface,image_list_sp[i],res,)))

    for p in predict_p:
        p.start()
    for p in predict_p:
        p.join()
    
    print(F'error: {res[0]}, total: {res[1]}')

if __name__=='__main__':
    # st = time.time()
    main()
    # print(time.time()-st)