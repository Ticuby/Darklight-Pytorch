import cv2
import os
path='../ARID_predict_farmes/'
path_dir = path
vedio_names = os.listdir()[0:-1]
for name in vedio_names:
    cap = cv2.VideoCapture(name)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    vedio_path=os.path.join(path_dir,name[:-4])
    i=1
    if not os.path.exists(vedio_path):
        os.mkdir(vedio_path)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("视频 %s 处理完成"%name)
            break
        cv2.imwrite(vedio_path+'\img_{0:05d}.jpg'.format(i),frame)
        i+=1
    # 完成所有操作后，释放捕获器
    cap.release()