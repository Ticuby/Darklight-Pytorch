import os
import csv
def csv_deal(csv_file:str,csv_type):
    path='../../ARID_frames/'
    with open(csv_file, newline="") as split_f:
        reader = csv.DictReader(split_f)   #把第一行作为key值
        save_txt=csv_type + '_' + "split1" + ".txt"
        with open(save_txt, 'w') as write_txt:
            for i,line in enumerate(reader):
                label = line["ClassID"]  # 视频种类
                name = line["Video"][:-4]  # 视频名称
                duration = str(len(os.listdir(path+name)))
                write_thing=name+' '+duration+' '+label+'\n'
                print("正在写入：视频："+name+'帧数：'+duration+'标签：'+label)
                write_txt.write(write_thing)

#csv_deal("ARID1.1_t1_train_pub.csv","train")
csv_deal("ARID1.1_t1_test_pub.csv","val")
#txt_deal('train_rgb_split1.txt')