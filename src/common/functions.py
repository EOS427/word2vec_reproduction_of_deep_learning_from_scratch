from .choose_proc import proc


def sigmoid(x):
    return 1 / (1 + proc.exp(-x))


def relu(x):
    return proc.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = proc.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - proc.max(x)
        x = proc.exp(x) / proc.sum(proc.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 在监督标签为one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -proc.sum(proc.log(y[proc.arange(batch_size), t] + 1e-7)) / batch_size
    #取出标签对应列元素
    #批处理返回平均值