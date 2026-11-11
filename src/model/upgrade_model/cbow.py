from ...common.choose_proc import proc
from ...common.accelerate_layers import Embedding,Negative_Sampling_Loss

class cbow:

    def __init__(self,voc_size,hid_size,win_size,corpus):
        W_in=0.01*proc.random.randn(voc_size,hid_size,).astype('f')#权重初始化
        W_out=0.01*proc.random.randn(voc_size,hid_size).astype('f')#负采样按词ID取行，必须是(词汇数,隐藏维)

        self.context_len=2*win_size#考虑到窗口变化，保存上下文大小供使用

        self.in_layer_list=[]#所有上下文相加求平均，embedding层初始化
        for i in range(self.context_len):
            self.in_layer_list.append(Embedding(W_in))

        self.ns_loss_layer=Negative_Sampling_Loss(W_out,
                                                  corpus,#使用ID列表统计概率
                                                  power=0.75,
                                                  sample_size=5)

        layer_list=self.in_layer_list+[self.ns_loss_layer]#设置参数和梯度
        self.params=[]
        self.grads=[]
        for layer in layer_list:
            self.params+=layer.params
            self.grads+=layer.grads

        self.word_vecs=W_in


    def forward(self,contexts,targets):#向前推进
        h=0#隐藏向量累加器，首次相加后形状自动为(batch_size,hid_size)

        for idx,embed_layer in enumerate(self.in_layer_list):
            h+=embed_layer.forward(contexts[:,idx])#按列取出分布式相加

        h/=self.context_len
        loss=self.ns_loss_layer.forward(targets,h)

        return loss


    def backward(self,dout=1):#反向传播
        dh=self.ns_loss_layer.backward(dout)
        dh/=self.context_len

        for layer in self.in_layer_list:
            layer.backward(dh)

        return None

