# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # 车辆分类
# %% [markdown]
# 使用迁移学习进行车辆分类

# %%
#安装 Paddlehub

# %% [markdown]
# 加载数据

# %%
import paddlehub.vision.transforms as T
from car_classification_dataset import CarsForClassification
transforms = T.Compose(
        [T.Resize((256, 256)),
         T.CenterCrop(224),
         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        to_rgb=True)

cars_train = CarsForClassification(transforms)
cars_test =  CarsForClassification(transforms, mode='test')


# %%
CarsForClassification()


# %%
import paddlehub as hub
import json
from car_classification_dataset import CAR_CLASSIFICATION_PATH

with open(CAR_CLASSIFICATION_PATH + "map.json") as f:
    dat = json.load(f)
model = hub.Module(name="resnet50_vd_imagenet_ssld", label_list=['SUV', 'small-sized truck', 'pickup truck', ' small-sized truck', 'tank car/tanker', 'others', 'minivan', 'bulk lorry/fence truck', 'light passenger vehicle', 'sedan', 'minibus', 'large-sized bus', 'business purpose vehicle/MPV', 'HGV/large truck'])


# %%
config = hub.RunConfig(
    use_cuda=True,                              #是否使用GPU训练，默认为False；
    num_epoch=10,                                #Fine-tune的轮数；
    checkpoint_dir="car_classification_model",#模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
    batch_size=128,                              #训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
    eval_interval=50,                           #模型评估的间隔，默认每100个step评估一次验证集；
    )  #Fine-tune优化策略；


# %%
import paddle
from paddlehub.finetune.trainer import Trainer
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
trainer = Trainer(model, optimizer, checkpoint_dir='car_classification_model', use_gpu=False)


# %%
trainer.train(cars_train, epochs=10, batch_size=64, eval_dataset=cars_test, save_interval=1)


# %%


5