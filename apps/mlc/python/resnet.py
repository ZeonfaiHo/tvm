#导入库
import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor



#加载onnx预训练模型
model_url = (
    "https://github.com/onnx/models/raw/main/"
    "vision/classification/resnet/model/"
    "resnet50-v2-7.onnx"
)

model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
onnx_model = onnx.load(model_path)

#读取图片
import os
from PIL import Image
import requests

img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
local_img_path = "imagenet_cat.png"

# 检查图片是否存在本地
def check_and_download_image(url, local_path):
    if not os.path.exists(local_path): # 图片不存在
        print("图片不存在，正在下载...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print("图片下载并保存成功")
        else:
            print("无法下载图片，状态码:", response.status_code)
    else:
        print("图片已存在")


check_and_download_image(img_url, local_img_path)
resized_image = Image.open(local_img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# Our input image is in HWC layout while ONNX expects CHW input, so convert the array
img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to the ImageNet input specification
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
img_data = np.expand_dims(norm_img_data, axis=0)

#Import the graph to Relay
# 将onnx 图转换为Relay图，Input name是随意的
# The input name may vary across model types. You can use a tool
# like Netron to check input names
targets = {
    "cpu": "llvm",
    "cuda": "cuda"
}
input_name = "data"
shape_dict = {input_name: img_data.shape}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
# print('Raw module:\n' + str(mod))

#标注device
cached_expr = dict()

def annotate_device(expr: relay.Expr):
    """递归遍历表达式并标注所有conv2d算子"""
    if expr in cached_expr:
        return cached_expr[expr]

    print(type(expr))

    if isinstance(expr, relay.Call):
        # 为conv2d算子添加CUDA注解
        new_args = [annotate_device(arg) for arg in expr.args]
        new_expr = relay.Call(expr.op, new_args, expr.attrs)
        if expr.op.name in ["nn.conv2d", 'nn.relu', 'nn.batch_norm']:
            new_expr = relay.annotation.on_device(new_expr, tvm.cuda())
        else:
            new_expr = relay.annotation.on_device(new_expr, tvm.cpu())
    elif isinstance(expr, relay.expr.TupleGetItem):
        expr: relay.expr.TupleGetItem
        new_value = annotate_device(expr.tuple_value)
        new_expr = relay.expr.TupleGetItem(new_value, expr.index)
    else:
        new_expr = expr
    
    cached_expr[expr] = new_expr
    return new_expr

# 获取main函数
main_func = mod["main"]
main_func_expr = main_func.body
print('Original main function:\n' + str(main_func_expr))

new_main_func = annotate_device(main_func_expr)
print('New main function:\n' + str(new_main_func))
mod = tvm.IRModule.from_expr(new_main_func)

print('Transformed IRModule: ' + str(mod))

# 编译
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target=targets, params=params)

module = graph_executor.GraphModule(lib["default"](tvm.cpu(), tvm.cuda()))

##% TVM上执行可移植图(Execute the portable graph on TVM)
# 现在，尝试在目标上部署已编译的模型
dtype = "float32"
module.set_input(input_name, img_data)
# while True:
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

#统计基本性能数据
import timeit

timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)

#%% 输出后处理，将TVM输出结果转化为更易读的结果
#其中的imagenet_synsets.txt和imagenet_classes.txt两个文件，url链接出错的话可自行下载：
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/data/ 下
from scipy.special import softmax

# Download a list of labels
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

# Open the output and read the output tensor
scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))