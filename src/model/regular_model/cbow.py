# import numpy as np

from ...common.choose_proc import proc

from ...common.layers import MatMul,SoftmaxWithLoss

class cbow:

    def __init__(self,voc_size,hid_size):
        #随机生成权重
        self.W_in=0.01*proc.random.randn(voc_size,hid_size).astype("f")
        self.W_out=0.01*proc.random.randn(hid_size,voc_size).astype("f")

        #设置层
        self.in_layer0= MatMul(self.W_in)
        self.in_layer1=MatMul(self.W_in)
        self.out_layer=MatMul(self.W_out)
        self.loss_layer=SoftmaxWithLoss()

        #设置选取分布式
        self.word_vecs=self.W_in

        #配置参数和梯度
        self.params=[]
        self.grads=[]
        layer_list=[self.in_layer0,self.in_layer1,self.out_layer]
        for layer in layer_list:#普通列表嵌套各层参数/梯度np矩阵，详见层定义
            self.params+=layer.params
            self.grads+=layer.grads#共享参数被重复放入，后期需处理，详见trainer


    def forward(self,contexts,targets):#向前推进
        context0=contexts[:,0]
        context1=contexts[:,1]
        out0=self.in_layer0.forward(context0)
        out1=self.in_layer1.forward(context1)

        average=(out0+out1)/2
        score=self.out_layer.forward(average)

        loss=self.loss_layer.forward(score,targets)

        return loss


    def backward(self,dout=1):#向后传递
        ds=self.loss_layer.backward(dout)
        da=self.out_layer.backward(ds)
        da*=0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)

        return None

