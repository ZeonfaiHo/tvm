"""
读取测试数据
"""
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
import tvm

mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

# 定义模型函数
import tvm.relay as relay

type_annotation = relay.TensorType(shape=(1, 10), dtype="float32")
def build_mnist_func(data: relay.Var,
                     w0: relay.Constant,
                     b0: relay.Constant,
                     w1: relay.Constant,
                     b1: relay.Constant):
    lv0 = relay.nn.matmul(data, w0, transpose_b=True)
    # lv0 = relay.annotation.on_device(lv0, tvm.cuda())
    lv1 = relay.nn.bias_add(lv0, b0)
    lv2 = relay.nn.relu(lv1)
    lv3 = relay.nn.matmul(lv2, w1, transpose_b=True)
    # lv3 = relay.annotation.function_on_device(lv3, tvm.cuda())
    lv4 = relay.nn.bias_add(lv3, b1)
    output = lv4

    return relay.Function([data], output, ret_type=type_annotation)

# 创建模型
mnist_func = build_mnist_func(
    relay.var("data", shape=(1, 784)),
    relay.Constant(tvm.nd.array(nd_params['w0'])),
    relay.Constant(tvm.nd.array(nd_params['b0'])),
    relay.Constant(tvm.nd.array(nd_params['w1'])),
    relay.Constant(tvm.nd.array(nd_params['b1']))
)

# 创建全局变量和模块
add_gvar_main = relay.GlobalVar("main")
mod = tvm.IRModule({add_gvar_main: mnist_func})

#标注device
def annotate_device(expr: relay.Expr):
    """递归遍历表达式并标注所有conv2d算子"""
    if isinstance(expr, relay.Call):
        # 为conv2d算子添加CUDA注解
        new_args = [annotate_device(arg) for arg in expr.args]
        new_expr = relay.Call(expr.op, new_args, expr.attrs)
        if expr.op.name == "nn.relu":
            return relay.annotation.on_device(new_expr, tvm.cuda())
        else:
            return relay.annotation.on_device(new_expr, tvm.cpu())
    else:
        return expr

# 获取main函数
main_func = mod["main"]
main_func_expr = main_func.body
print('Original main function:\n' + str(main_func_expr))

# 标注device
new_main_func = annotate_device(main_func_expr)
print('New main function:\n' + str(new_main_func))
mod = tvm.IRModule.from_expr(new_main_func)

print('Transformed IRModule: ' + str(mod))

# 定义目标
targets = {
    "cpu": "llvm",
    "cuda": "cuda"
}

# 构建模型
# with tvm.transform.PassContext(opt_level=0, disabled_pass=['FuseOps', ]):
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target=targets)

# 运行

rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cpu(), tvm.cuda()))

rt_mod.set_input("data", tvm.nd.array(img.astype("float32")))
# while True:
rt_mod.run()
out = rt_mod.get_output(0)

# 输出

out_np = out.asnumpy()
print('Output: ' + str(out_np))
pred = out_np.argmax()
print(f"Prediction: {class_names[pred]}")