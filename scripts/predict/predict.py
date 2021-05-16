import os, sys
import numpy as np

import time
import argparse

import csv
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix

print(torch.cuda.is_available())
# datasetFolder="../../datasets"
# sys.path.insert(0, "../../")
import models
from scripts.eval.VideoSpatialPrediction3D_attention import VideoSpatialPrediction3D_attention

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch darklight')

parser.add_argument('--dataset', '-d', default='ARID',
                    choices=["ARID", "ucf101", "hmdb51"],
                    help='dataset: ARID | ucf101 | hmdb51')
parser.add_argument('--arch', '-a', metavar='ARCH', default='dark_light',  # default='dark_light',
                    choices=model_names)

parser.add_argument('-n', '--name', default='predict', type=str,
                    help='predict_fileName:predict.cvs')

parser.add_argument('--both-flow', default='True',
                    help='give dark and light flow both')

parser.add_argument('-g', '--gamma', default=2, type=float,
                    help="the value of gamma")

parser.add_argument('--no-attention', default=True, action='store_false', help="use attention to instead of linear")

multiGPUTest = False
multiGPUTrain = False
ten_crop_enabled = True
num_seg = 16
num_seg_3D = 1

result_dict = {}


def buildModel(model_path, num_categories):
    model = models.__dict__[args.arch](num_classes=num_categories, length=num_seg, both_flow=args.both_flow)
    params = torch.load(model_path)

    if multiGPUTest:
        model = torch.nn.DataParallel(model)
        new_dict = {"module." + k: v for k, v in params['state_dict'].items()}
        model.load_state_dict(new_dict)

    elif multiGPUTrain:
        new_dict = {k[7:]: v for k, v in params['state_dict'].items()}
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval()
    return model


def main():
    global args
    args = parser.parse_args()
    length = 64

    model_path = 'model_best.pth.tar'
    if not args.no_attention:
        args.arch = 'dark_light_noAttention'

    if args.dataset == 'ucf101':
        frameFolderName = "ucf101_predict_frames"
    elif args.dataset == 'hmdb51':
        frameFolderName = "hmdb51_predict_frames"
    elif args.dataset == 'ARID':
        frameFolderName = "ARID_predict_farmes"
    data_dir = os.path.join(frameFolderName)

    extension = 'img_{0:05d}.jpg'

    val_file = args.name + '.csv'

    start_frame = 0
    if args.dataset == 'ucf101':
        num_categories = 101
    elif args.dataset == 'hmdb51':
        num_categories = 51
    elif args.dataset == 'ARID':
        num_categories = 11

    model_start_time = time.time()
    spatial_net=buildModel(model_path,num_categories)
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))
    with open(val_file, newline="") as f:
        csv_list = f.readlines()

    f_val = open(val_file, newline="")
    reader = csv.DictReader(f_val)  # 把第一行作为key值
    print("we got %d test videos" % (len(csv_list)-1))

    line_id = 1
    match_count = 0
    match_count_top3 = 0

    y_true = []
    y_pred = []
    timeList = []
    # result_list = []
    field_names = ['VideoID', 'Video', 'ClassID']
    pred_rows = []

    for i, line in enumerate(reader):
        vdeio_id = int(line["VideoID"])
        vdeio_name = str(line["VideoID"])+".mp4"
        vdeio_dir = line["VideoID"]
        clip_path = os.path.join(data_dir, vdeio_dir)
        duration = line["duration"] #帧长
        input_video_label = None

        start = time.time()

        spatial_prediction = VideoSpatialPrediction3D_attention(
            clip_path,
            spatial_net,
            num_categories,
            args.arch,
            start_frame,
            duration,
            num_seg=num_seg_3D,
            length=length,
            extension=extension,
            ten_crop=ten_crop_enabled,
            gamma=args.gamma)

        end = time.time()
        estimatedTime = end - start
        timeList.append(estimatedTime)

        pred_index, mean_result, top3 = spatial_prediction


        pred_row = {field_names[0]: vdeio_id, field_names[1]: vdeio_name, field_names[2]: pred_index}
        pred_rows.append(pred_row)

        print("Sample %d/%d: Prediction: %d" % (line_id, len(csv_list)-1, pred_index))
        print("Estimated Time  %0.4f" % estimatedTime)
        print("------------------")

        line_id += 1
        y_pred.append(pred_index)

    with open('predict_result.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for pred_row in pred_rows:
            writer.writerow(pred_row)


if __name__ == "__main__":
    main()
