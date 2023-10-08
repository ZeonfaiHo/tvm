"""
读取测试数据
"""
import tvm.contrib.graph_executor
import tvm.relay
from tvm import IRModule
import tvm
import pickle as pkl
import torchvision
import torch.utils.data
import time


test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=1, shuffle=False)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

img, label = next(iter(test_loader))
img = img.reshape(1, 28, 28).numpy()


"""
读取模型权重
"""
mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}


"""
构建relay模型
"""
type_annotation = tvm.relay.TensorType(shape=(1, 10), dtype="float32")
def build_mnist_func(data: tvm.relay.Var,
                     w0: tvm.relay.Constant,
                     b0: tvm.relay.Constant,
                     w1: tvm.relay.Constant,
                     b1: tvm.relay.Constant):
    lv0 = tvm.relay.nn.matmul(data, w0, transpose_b=True)
    lv1 = tvm.relay.nn.bias_add(lv0, b0)
    lv2 = tvm.relay.nn.relu(lv1)
    lv3 = tvm.relay.nn.matmul(lv2, w1, transpose_b=True)
    lv4 = tvm.relay.nn.bias_add(lv3, b1)
    output = lv4

    return tvm.relay.Function([data], output, ret_type=type_annotation)


mnist_func = build_mnist_func(tvm.relay.var("data", shape=(1, 784)),
                              tvm.relay.Constant(nd_params['w0']),
                              tvm.relay.Constant(nd_params['b0']),
                              tvm.relay.Constant(nd_params['w1']),
                              tvm.relay.Constant(nd_params['b1']))
add_gvar_main = tvm.relay.GlobalVar("main")
mod = tvm.IRModule({add_gvar_main: mnist_func})

"""
编译模型
"""

# lib = tvm.relay.build(mod, target='cuda')
# # lib.export_library('compiled_model.tar')
# rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cuda()))

# graph, lib, params = tvm.relay.build(mod, target='llvm')
lib = tvm.relay.build(mod, target='llvm')
lib.export_library('compiled_model.tar')
# print(lib.get_source())
rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cpu()))

rt_mod.set_input("data", tvm.nd.array(img.astype("float32")))
rt_mod.run()

out = rt_mod.get_output(0)
out_np = out.asnumpy()
print('Output: ' + str(out_np))
pred = out_np.argmax()
print(f"Prediction: {class_names[pred]}")
