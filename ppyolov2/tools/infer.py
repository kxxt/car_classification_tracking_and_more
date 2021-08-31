# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.image as mpimg
import paddlehub as hub
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('train')

import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",  # 命令行的参数名，用法： --infer_dir xxxx ，xxxx为你为这个参数添加的值
        type=str,       # 该参数的数据类型是string
        default=None,   # 默认值为None
        help="Directory for images to perform inference on.") # 对该参数解释说明
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="Whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    parser.add_argument(
        "--save_txt",
        type=bool,
        default=True,
        help="Whether to save inference result in txt.")
    parser.add_argument(
        "--weight",
        type=str,
        default="model_params/72.pdparams",
        help="Whether to save inference result in txt.")
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def get_right_bbox(bbox_lists):
    res = np.array([])
    for i in range(len(bbox_lists)):
        xmin, ymin, w, h = bbox_lists[i]['bbox']
        xmax = xmin + w
        ymax = ymin + h
        res = np.append(res,[xmin,ymin,xmax,ymax])
    res = res.reshape((-1,4))
    return res


def get_lisence_result(img, bboxs, ocr,adjust = 0.08):
    error = adjust
    result = []
    for i in range(len(bboxs)):
        xmin, ymin, xmax, ymax = bboxs[i]
        # 把车牌图片定位提取出来
        n_img = img[int(ymin*(1-error)):int(ymax*(1+error)),int(xmin*(1-error)):int(xmax*(1+error)),:]
        # cv2.imwrite("../n_img{}.png".format(i),n_img)
        # 进行ocr识别
        ocr_result = ocr.recognize_text(
                    images=[n_img],         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                    use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                    # output_dir='../data',  # 图片的保存路径，默认设为 ocr_result；
                    visualization=False,       # 是否将识别结果保存为图片文件；
                    box_thresh=0.2,           # 检测文本框置信度的阈值；
                    text_thresh=0.2)          # 识别中文文本置信度的阈值；
        res = ocr_result[0]['data']
        if res == []:
            result.append(None)
        else:
            result.append(res[0]['text'])
    return result



def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=24):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor,stroke_width=1, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_bbox(img, bboxs, recog_res=None):
    error=  0
    for i in range(len(bboxs)):
        xmin, ymin, xmax, ymax = bboxs[i]
        # print(xmin, ymin, xmax, ymax)
        # 画出方形框
        cv2.rectangle(
            img,
            (int(xmin*(1-error)),int(ymin*(1-error))),
            (int(xmax*(1+error)),int(ymax*(1+error))),
            (0, 255, 0),
            2
        )
        try:
            if recog_res[i] == None:
                continue
                # cv2.putText(
                #     img,'Unknown',
                #     (int(xmin*1),int(ymin*(1-error))),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,(0, 255, 0),2
                # )
            else:
            # print(recog_res[i])
            # 添加字
                res_string = recog_res[i]
                img = cv2ImgAddText(
                    img, res_string,
                    int(xmin),int(ymin-30),
                )
        except:
            continue
    return img

def write_result(filepath, index, bbox_lists,recog_res):
    with open(filepath,'a') as f:
        # 对识别出的车牌标定框循环写入
        for i in range(len(bbox_lists)):
            f.write(str(index)+",")
            # 对单个车牌的位置循环写入
            for j in range(len(bbox_lists[i]['bbox'])):
                f.write(str(bbox_lists[i]['bbox'][j])+",")
            f.write(str(1)+",")
            f.write(str(recog_res[i])+"\n")

# 检测车流视频
def run(FLAGS, cfg):
    # 加载移动端预训练模型
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")
    # build trainer
    trainer = Trainer(cfg, mode='test')
    # load weights
    trainer.load_weights(cfg.weights)
    # inference
    images_path = ["./temp.png"]
    # 读取视频
    cap = cv2.VideoCapture("./test2.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 要保存的视频格式
    # 获取总帧数
    frames_num=int(cap.get(7))
    current_frame = 1
    # 把处理过的视频保存下来
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 保存的视频地址
    video_save_path = './pp_yolo_2021080241745.mp4'
    output_viedo = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, fram = cap.read()  # 读取视频返回视频是否结束的bool值和每一帧的图像
        cv2.imwrite(images_path[0], fram) # 保存图片：路径+frame(img)
        bbox_lists = trainer.predict_get(images_path,draw_threshold=FLAGS.draw_threshold,output_dir=FLAGS.output_dir,save_txt=FLAGS.save_txt)
        current_bbox = get_right_bbox(bbox_lists)  # 返回车牌坐标信息的数组
        img = cv2.imread(images_path[0]) # 读取一帧图片
        recog_res = get_lisence_result(img, current_bbox, ocr, 0.08) # 识别车牌(ocr)
        print("{}/{}".format(current_frame,frames_num),end=" ")
        write_result('./det_results/20210824_2.txt', current_frame, bbox_lists, recog_res)
        print(recog_res)  # 可视化视频识别进度
        img = draw_bbox(img,current_bbox, recog_res)   # 标出车牌框，显示文字
        output_viedo.write(img)  # 把帧写入到视频中
        current_frame+=1

# 检测单张图片车牌
def run_photo(FLAGS, cfg):
    # 加载移动端预训练模型
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")
    # build trainer
    trainer = Trainer(cfg, mode='test')
    # load weights
    trainer.load_weights(cfg.weights)
    # inference
    images_path =['./21.jpg']
    img = cv2.imread(images_path[0])
    bbox_lists = trainer.predict_get(images_path,draw_threshold=FLAGS.draw_threshold,output_dir=FLAGS.output_dir,save_txt=FLAGS.save_txt)
    # print(bbox_lists)
    current_bbox = get_right_bbox(bbox_lists)
        # 读取测试文件夹test.txt中的照片路径
        # np_images =[cv2.imread(image_path) for image_path in imgages_path]
    recog_res = get_lisence_result(img, current_bbox,ocr,0.08)
    img = draw_bbox(img,current_bbox,recog_res)
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite('./result1.jpg',img)


def main():
    FLAGS = parse_args()
    # FLAGS包含先前parse_args中的所有参数，参数值要么为默认值，要么为命令行传入参数
    # 后面强制更改参数的值
    FLAGS.config = "./configs/ppyolo/ppyolov2_r50vd_dcn_voc.yml"
    cfg = load_config(FLAGS.config)
    cfg.weights = "./models/19.pdparams"
    cfg['use_vdl'] = FLAGS.use_vdl
    cfg['vdl_log_dir'] = FLAGS.vdl_log_dir
    merge_config(FLAGS.opt)

    place = paddle.set_device('gpu' if cfg.use_gpu else 'cpu')

    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
        cfg['norm_type'] = 'bn'

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    return FLAGS,cfg

    # run_photo(FLAGS, cfg)


# 图片接口
def reco_photo(img_path):
    FLAGS,cfg = main()
    # 加载移动端预训练模型
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")
    # build trainer
    trainer = Trainer(cfg, mode='test')
    # load weights
    trainer.load_weights(cfg.weights)
    # inference
    images_path =[img_path]
    img = cv2.imread(images_path[0])
    bbox_lists = trainer.predict_get(images_path,draw_threshold=FLAGS.draw_threshold,output_dir=FLAGS.output_dir,save_txt=FLAGS.save_txt)
    # print(bbox_lists)
    current_bbox = get_right_bbox(bbox_lists)
    # 读取测试文件夹test.txt中的照片路径
    # np_images =[cv2.imread(image_path) for image_path in imgages_path]
    recog_res = get_lisence_result(img, current_bbox,ocr,0.08)
    img = draw_bbox(img, current_bbox, recog_res)
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite('./result1.jpg', img)

# 视频接口
def reco_vedio(path, sv_txt):
    FLAGS,cfg = main()
    # 加载移动端预训练模型
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")
    # build trainer
    trainer = Trainer(cfg, mode='test')
    # load weights
    trainer.load_weights(cfg.weights)
    # inference
    images_path = ["./temp.png"]
    # 读取视频
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 要保存的视频格式
    # 获取总帧数
    frames_num=int(cap.get(7))
    current_frame = 1
    # 把处理过的视频保存下来
    output_viedo = cv2.VideoWriter()
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 保存的视频地址
    video_save_path = './pp_yolo_2021080241745.mp4'
    output_viedo = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, fram = cap.read()  # 读取视频返回视频是否结束的bool值和每一帧的图像
        cv2.imwrite(images_path[0], fram) # 保存图片：路径+frame(img)
        bbox_lists = trainer.predict_get(images_path,draw_threshold=FLAGS.draw_threshold,output_dir=FLAGS.output_dir,save_txt=FLAGS.save_txt)
        current_bbox = get_right_bbox(bbox_lists)  # 返回车牌坐标信息的数组
        img = cv2.imread(images_path[0]) # 读取一帧图片
        recog_res = get_lisence_result(img, current_bbox, ocr, 0.08) # 识别车牌(ocr)
        print("{}/{}".format(current_frame,frames_num),end=" ")
        if(sv_txt):
            write_result(sv_txt, current_frame, bbox_lists, recog_res)
        print(recog_res)  # 可视化视频识别进度
        img = draw_bbox(img,current_bbox, recog_res)   # 标出车牌框，显示文字
        output_viedo.write(img)  # 把帧写入到视频中
        current_frame+=1




if __name__ == '__main__':
    # main()
    reco_photo('./21.jpg')  # 图片车牌识别
    # reco_vedio("./test2.mp4",'./det_results/res.txt' ) # 图片视频识别

