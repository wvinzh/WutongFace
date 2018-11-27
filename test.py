from wutong_face import WutongFace

wface = WutongFace()
image_file = "F:\\google_images\\蔡奇\\000002.jpg"
faces_no_align = wface.face_no_align(image_file,crop_size=(180,180),min_face_size=10)
faces_no_align[0].show()