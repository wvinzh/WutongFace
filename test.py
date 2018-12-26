from wutong_face import WutongFace
import os
import time
def test():
    nnn_txt = "F:\\Test_N.txt"
    ppp_txt = "F:\\Test_P.txt"
    root = "F:\\TestSet"
    r = 0
    with open(nnn_txt,'r',encoding='utf-8') as f:
        normals = f.readlines()
        normals = [os.path.join(root,n.strip().rsplit(' ',1)[0]) for n in normals]
    with open(ppp_txt,'r',encoding='utf-8') as f:
        politicals = f.readlines()
        politicals = [os.path.join(root,n.strip().rsplit(' ',1)[0]) for n in politicals]
    # test political
    wface = WutongFace()
    n_total = len(normals)
    r = n_total
    print('total %d' % r)
    for pimage in normals:
        # print(p)
        faces_align,_,_ = wface.face_align(pimage,min_face_size=10,crop_size=(150,150))
        if len(faces_align) <= 0: 
            continue
        features = wface.feature_extractor('resnet',faces_align)

        classifier,name_dict = wface.get_classifier()

        pred = classifier.predict(features)
        pl = 0
        for p in pred:
            if p in name_dict:
                pl+=1
                # print(name_dict[p])
        if pl <= 0:
            print(pimage)
            r-=1

    print(F'right: {r}, total: {n_total}, acc: {float(r/n_total)}')
        

if __name__=='__main__':
    st = time.time()
    test()
    print(time.time()-st)