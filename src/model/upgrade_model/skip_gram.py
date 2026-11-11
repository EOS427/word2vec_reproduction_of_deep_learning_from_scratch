from ...common.choose_proc import proc
from ...common.accelerate_layers import Embedding,Negative_Sampling_Loss

class skip_gram:

    def __init__(self,voc_size,hid_size,win_size,corpus):
        W_in=0.01*proc.random.randn(voc_size,hid_size).astype('f')
        W_out=0.01*proc.random.randn(voc_size,hid_size).astype('f')#负采样按词ID取行，必须是(词汇数,隐藏维)

        self.context_len=2*win_size

        self.in_layer=Embedding(W_in)
        self.loss_layer_list=[]
        for i in range(self.context_len):
            self.loss_layer_list.append(Negative_Sampling_Loss(W_out,
                                                               corpus,
                                                               power=0.75,
                                                               sample_size=5))
        self.params=[]
        self.grads=[]
        layers=[self.in_layer]+self.loss_layer_list
        for layer in layers:
            self.params+= layer.params
            self.grads+= layer.grads

        self.word_vecs=W_in


    def forward(self,context,target):
        h=self.in_layer.forward(target)

        loss=0
        for idx,ns_layer in enumerate(self.loss_layer_list):
            loss+=ns_layer.forward(context[:,idx],h)

        return loss


    def backward(self,dout=1):
        dh=0
        for ns_layer in self.loss_layer_list:
            dh+=ns_layer.backward(dout)

        self.in_layer.backward(dh)

        return None
