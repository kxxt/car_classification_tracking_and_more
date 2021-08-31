from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import paddlehub as hub
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
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",  # 命令行的参数名，用法： --infer_dir xxxx ，xxxx为你为这个参数添加的值
        type=str,       # 该参数的数据类型是string
        default=[],   # 默认值为None
        help="Directory for images to perform inference on.") # 对该参数解释说明
    parser.add_argument(
        "--infer_img",
        type=str,
        default=[],
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=[],
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--slim_config",
        default=[],
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
        default="models/19.pdparams",
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


def get_lisence_result(img, bboxs, ocr, adjust = 0.08, is_gpu = False):
    error = adjust
    result = []
    for i in range(len(bboxs)):
        xmin, ymin, xmax, ymax = bboxs[i]
        # 把车牌图片定位提取出来
        n_img = img[int(ymin*(1-error)):int(ymax*(1+error)),int(xmin*(1-error)):int(xmax*(1+error)),:]
        # cv2.imwrite("../n_img{}.png".format(i),n_img)
        # 进行ocr识别
        ocr_result = ocr.recognize_text(
                    images=[n_img],           # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                    use_gpu=is_gpu,           # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
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
            else:
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

# 检测单张图片车牌
def run_img(FLAGS, cfg, img_path, save_path):
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
    current_bbox = get_right_bbox(bbox_lists)
    recog_res = get_lisence_result(img, current_bbox,ocr,0.08)
    img = draw_bbox(img,current_bbox,recog_res)
    if not os.path.exists('ppvolo_res'):
        os.mkdir('ppvolo_res')
    cv2.imwrite(os.path.join('ppvolo_res',save_path), img)




def main_img(
        img_path,
        weights = "./models/19.pdparams",
        img_save_path = "result.jpg",
):
    FLAGS = parse_args()
    # FLAGS包含先前parse_args中的所有参数，参数值要么为默认值，要么为命令行传入参数
    # 后面强制更改参数的值
    FLAGS.config = "./configs/ppyolo/ppyolov2_r50vd_dcn_voc.yml"
    cfg = load_config(FLAGS.config)
    cfg['use_vdl'] = FLAGS.use_vdl
    cfg['vdl_log_dir'] = FLAGS.vdl_log_dir
    cfg.weights = weights

    merge_config(FLAGS.opt)

    place = paddle.set_device('gpu' if cfg.use_gpu else 'cpu')

    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
        cfg['norm_type'] = 'bn'

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    run_img(FLAGS,cfg,img_path,img_save_path)


if __name__ == '__main__':
    main_img('test.png')