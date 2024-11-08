# 完成必要的import（下文省略）
from cn_clip.deploy.tensorrt_utils import TensorRTModel
from PIL import Image
import numpy as np
import torch
import argparse
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from cn_clip.clip.utils import _MODELS, _MODEL_INFO, _download, available_models, create_model, image_transform

# 载入TensorRT图像侧模型（**请替换${DATAPATH}为实际的路径**）
img_trt_model_path="D:/projects/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14-336px/deploy/vit-l-14-336.img.fp16.trt"
img_trt_model = TensorRTModel(img_trt_model_path)

# 预处理图片
model_arch = "ViT-L-14-336" # 这里我们使用的是ViT-B-16规模，其他规模请对应修改
preprocess = image_transform(_MODEL_INFO[model_arch]['input_resolution'])
# 示例皮卡丘图片，预处理后得到[1, 3, 分辨率, 分辨率]尺寸的Torch Tensor
image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).cuda()

# 用TensorRT模型计算图像侧特征
image_features_trt = img_trt_model(inputs={'image': image})['unnorm_image_features'] # 未归一化的图像特征
image_features_trt /= image_features_trt.norm(dim=-1, keepdim=True) # 归一化后的Chinese-CLIP图像特征，用于下游任务
print(image_features_trt.shape) # Torch Tensor shape: [1, 特征向量维度]



# # 载入TensorRT文本侧模型（**请替换${DATAPATH}为实际的路径**）
# txt_trt_model_path="D:/projects/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14-336px/deploy/vit-l-14-336.txt.fp16.trt"
# txt_trt_model = TensorRTModel(txt_trt_model_path)

# # 为4条输入文本进行分词。序列长度指定为52，需要和转换ONNX模型时保持一致（参见ONNX转换时的context-length参数）
# text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], context_length=52).cuda()


# # 用TensorRT模型依次计算文本侧特征
# text_features_trt = []
# for i in range(len(text)):
#     # 未归一化的文本特征
#     text_feature_trt = txt_trt_model(inputs={'text': torch.unsqueeze(text[i], dim=0)})['unnorm_text_features']
#     text_features_trt.append(text_feature_trt)
# text_features_trt = torch.squeeze(torch.stack(text_features_trt), dim=1) # 4个特征向量stack到一起
# text_features_trt = text_features_trt / text_features_trt.norm(dim=1, keepdim=True) # 归一化后的Chinese-CLIP文本特征，用于下游任务
# print(text_features_trt.shape) # Torch Tensor shape: [4, 特征向量维度]









# 完成必要的import（下文省略）
import onnxruntime

# 载入ONNX图像侧模型（**请替换${DATAPATH}为实际的路径**）
img_sess_options = onnxruntime.SessionOptions()
img_run_options = onnxruntime.RunOptions()
img_run_options.log_severity_level = 2
img_onnx_model_path="D:/projects/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14-336px/deploy/vit-l-14-336.img.fp16.onnx"
img_session = onnxruntime.InferenceSession(img_onnx_model_path,
                                        sess_options=img_sess_options,
                                        providers=["CUDAExecutionProvider"])

# 用ONNX模型计算图像侧特征
image_features_onnx = img_session.run(["unnorm_image_features"], {"image": image.cpu().numpy()})[0] # 未归一化的图像特征
image_features_onnx = torch.tensor(image_features_onnx)
image_features_onnx /= image_features_onnx.norm(dim=-1, keepdim=True) # 归一化后的Chinese-CLIP图像特征，用于下游任务
print(image_features_onnx.shape) # Torch Tensor shape: [1, 特征向量维度]




# 载入ONNX文本侧模型（**请替换${DATAPATH}为实际的路径**）
txt_sess_options = onnxruntime.SessionOptions()
txt_run_options = onnxruntime.RunOptions()
txt_run_options.log_severity_level = 2
txt_onnx_model_path="D:/projects/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14-336px/deploy/vit-l-14-336.txt.fp16.onnx"
txt_session = onnxruntime.InferenceSession(txt_onnx_model_path,
                                        sess_options=txt_sess_options,
                                        providers=["CUDAExecutionProvider"])

# 为4条输入文本进行分词。序列长度指定为52，需要和转换ONNX模型时保持一致（参见转换时的context-length参数）
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], context_length=52) 


# 用ONNX模型依次计算文本侧特征
text_features_onnx = []
for i in range(len(text)):
    one_text = np.expand_dims(text[i].cpu().numpy(),axis=0)
    text_feature_onnx = txt_session.run(["unnorm_text_features"], {"text":one_text})[0] # 未归一化的文本特征
    text_feature_onnx = torch.tensor(text_feature_onnx)
    text_features_onnx.append(text_feature_onnx)
text_features_onnx = torch.squeeze(torch.stack(text_features_onnx),dim=1) # 4个特征向量stack到一起
text_features_onnx = text_features_onnx / text_features_onnx.norm(dim=1, keepdim=True) # 归一化后的Chinese-CLIP文本特征，用于下游任务
print(text_features_onnx.shape) # Torch Tensor shape: [4, 特征向量维度]




print(100 * image_features_trt.to('cpu') @ image_features_onnx.t())
print((100 * image_features_trt.to('cpu') @ text_features_onnx.t()).softmax(dim=-1))
print((100 * image_features_onnx @ text_features_onnx.t()).softmax(dim=-1))
# for i in range(len(text_features_trt)):
#     print(100 * text_features_trt[i].to('cpu') @ text_features_onnx[i].t())