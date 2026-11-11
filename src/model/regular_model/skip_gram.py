# import numpy as np

from ...common.choose_proc import proc

from ...common.layers import MatMul,SoftmaxWithLoss

class skip_gram:

    def __init__(self,voc_size,hid_size):
        #随机生成权重
        self.W_in=0.01*proc.random.randn(voc_size,hid_size).astype("f")
        self.W_out=0.01*proc.random.randn(hid_size,voc_size).astype("f")

        #设置层
        self.in_layer= MatMul(self.W_in)
        self.out_layer=MatMul(self.W_out)
        self.loss_layer0=SoftmaxWithLoss()
        self.loss_layer1=SoftmaxWithLoss()

        #设置选取分布式
        self.word_vecs=self.W_in

        #配置参数和梯度
        self.params=[]
        self.grads=[]
        layer_list=[self.in_layer,self.out_layer]
        for layer in layer_list:#普通列表嵌套各层参数/梯度np矩阵，详见层定义
            self.params+=layer.params
            self.grads+=layer.grads#共享参数被重复放入，后期需处理，详见trainer


    def forward(self,contexts,targets):#向前推进
        out=self.in_layer.forward(targets)
        score=self.out_layer.forward(out)

        context0=contexts[:,0]
        context1=contexts[:,1]
        loss0=self.loss_layer0.forward(score,context0)
        loss1=self.loss_layer1.forward(score,context1)
        loss=loss0+loss1

        return loss


    def backward(self,ds=1):#向后传递
        dloss0=self.loss_layer0.backward(ds)
        dloss1=self.loss_layer1.backward(ds)
        dloss=dloss0+dloss1

        dout=self.out_layer.backward(dloss)

        self.in_layer.backward(dout)

        return None