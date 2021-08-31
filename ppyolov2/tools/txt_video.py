import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np


def get_right_bbox(target_df, cur_frame):
    df = target_df[target_df['frame_id'] == cur_frame]
    # print(df)
    bbox_lists = df.values[:, 1:5]
    recog_res = df.values[:,-1]
    res = np.array([])
    # print(len(bbox_lists))
    for i in range(len(bbox_lists)):
        xmin, ymin, w, h = bbox_lists[i]
        xmax = xmin + w
        ymax = ymin + h
        res = np.append(res,[xmin,ymin,xmax,ymax])
    res = res.reshape((-1,4))
    return res,recog_res


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
            if recog_res[i] == 'None':
                continue
            else:
                res_string = recog_res[i]
                # print('img1:',img)
                img = cv2ImgAddText(
                    img, res_string,
                    int(xmin),int(ymin-30),
                )
                # print('img2',img)
        except:
            continue
    return img



def generate_result(recog_df,video_path,out_path):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    frame_num = video.get(7)# 这个get fps没啥用，不用来显示的

    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 要保存avi视频格式
    # output_viedo = cv2.VideoWriter('final.avi', fourcc, fps, (width, height))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 要保存mp4视频格式
    output_viedo = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    cur_frame = 1

    while True:
        ret, frame = video.read()
        if ret == True:
            print('{}/{}'.format(cur_frame,int(frame_num)))
            bbox_list, recog_res = get_right_bbox(recog_df,cur_frame)
            if bbox_list == []:
                continue
            img = draw_bbox(frame, bbox_list, recog_res)
            output_viedo.write(img)
            cv2.imwrite('./res_photo/{}.png'.format(str(cur_frame).zfill(6)),img)
            cur_frame+=1
        else:
            break


if __name__ == '__main__':
    target_df = pd.read_csv('./ppyolo_res/try_detail.txt')
    # 去掉目标检测的置信度
    target_df = target_df.drop(['score'],axis=1)
    # print(target_df)
    generate_result(target_df,'car.mp4','./output_.mp4')

