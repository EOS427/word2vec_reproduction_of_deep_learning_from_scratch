from PIL.features import version_codec
import numpy

from .choose_proc import GPU,proc
from .functions import cross_entropy_error
from collections import Counter


class Embedding:

    def __init__(self,W):
        self.params=[W]
        self.grads=[proc.zeros_like(W)]
        self.idx=None

    def forward(self,idx):
        W=self.params[0]#注意嵌套
        self.idx=idx
        vec=W[idx]
        return vec

    def backward(self,dout):
        dW=self.grads[0]
        dW[...]=0

        for i,ID in enumerate(self.idx):
            dW[ID]+=dout[i]

        return None
        #Embedding 的前向是“按索引从权重矩阵里取行”
        # 反向就是“按索引把上游梯度加回对应的行”
        # 输入是整数索引，不可导
        # 所以它不往前传梯度，只算自己权重 W 的梯度



class Embedding_Dot:#二分类取目标分布式并dot向量复合层

    def __init__(self,W):
        self.embed=Embedding(W)

        self.params=self.embed.params
        self.grads=self.embed.grads

        self.cache=None


    def forward(self,ID,vec):
        target_W=self.embed.forward(ID)#按词ID取出目标分布式，同时记录索引供反向传播
        score=proc.sum(target_W*vec,axis=1)
        self.cache=(vec,ID,target_W)

        return score


    def backward(self,dout):
        vec,ID,target_W=self.cache

        dout=dout.reshape(dout.shape[0],1)#竖起来
        dtarget_W=dout*vec
        self.embed.backward(dtarget_W)

        dvec=dout*target_W
        return dvec



class Sigmoid_With_Loss:#二分类sigmoid损失复合层

    def __init__(self):
        self.x=None
        self.y=None
        self.t=None
        self.loss=None

    def forward(self,x,t):
        self.x=x
        self.y=1/(1+proc.exp(-x))

        self.t=t
        self.loss=cross_entropy_error(proc.stack((1-self.y,self.y),axis=1),t)
        #将为0和为1的分数按列拼接在一起传入（stack对numpy/cupy均可用）
        #按标签抽出分数计算损失
        return self.loss

    def backward(self,dout=1):
        batch_size=self.x.shape[0]
        dx=(self.y-self.t)/batch_size#取平均

        return dx



class Unigram_Sampler:
    #数组全程numpy，避开兼容问题

    def __init__(self,sample_size,corpus,power):
        self.counter=Counter(corpus)#创建counter实例
        self.sample_size=sample_size

        self.voc_size=len(self.counter)
        self.percentage=proc.zeros(self.voc_size)
        for ID in corpus:
            self.percentage[ID]=self.counter[ID]#按序取出数量放入列表

        self.percentage=proc.power(self.percentage,power)#幂运算
        self.percentage=self.percentage/numpy.sum(self.percentage)#归一化


    def get_negative_sample(self,target):


        batch_size=target.shape[0]
        negative_samples=numpy.zeros((batch_size,self.sample_size),dtype=int)#采样结果将用作索引，必须是整数
        #源码里有一行特意写的注释"Refuse assignment from CPU arrays"拒绝接收 CPU 数组，见 cupy issue gh-2079
        # 它只肯收 cupy 数组或标量

        for idx in range(len(target)):
            prep_p=self.percentage.copy()#创建预处理概率

            ID=target[idx]
            prep_p[ID]=0#把正例自身的概率清零（负例不能是正例自己）
            prep_p/=prep_p.sum()#去除一个元素后综合不为一需要重新计算概率


            #agent查阅了cupy 官方文档和最新版（v14.x）源码确认：
            #cupy的`random.choice`从v10到最新的v14全版本都不支持"同时指定`replace=False`和非均匀概率`p` "
            #官方文档也注明了 "Currently`p` is not supported when`replace=False` "
            #所以内部全部改用numpy只在返回时转一次：
            negative_samples[idx,:]=numpy.random.choice(#不放回抽取,放入对应行
                self.voc_size,
                size=self.sample_size,
                replace=False,
                p = prep_p.get() if GPU else prep_p#区分兼容问题，方法不同
            )
            negative_samples=numpy.asarray(negative_samples)

        return negative_samples

#----------------------------------------------------------------------
#汇总上述类构造负采样正负例一体训练层
class Negative_Sampling_Loss:

    def __init__(self,W,corpus,power,sample_size):
        self.sample_size=sample_size
        self.sampler=Unigram_Sampler(sample_size,corpus,power)

        self.loss_layer=[Sigmoid_With_Loss() for i in range(sample_size+1)]
        self.embed_dot_layer=[Embedding_Dot(W) for i in range(sample_size+1)]

        self.params=[]
        self.grads=[]
        for layer in self.embed_dot_layer:
            self.params+=layer.params
            self.grads+=layer.grads


    def forward(self,target,h):
        batch_size=len(target)
        loss=0#正负例损失之和，各损失层返回标量，直接用标量累加

        negative_samples=self.sampler.get_negative_sample(target)

        correct_label=proc.ones(batch_size,dtype=int)#正例处理
        correct_score=self.embed_dot_layer[0].forward(target,h)
        loss+=self.loss_layer[0].forward(correct_score,correct_label)

        negative_label=proc.zeros(batch_size,dtype=int)#负例处理
        for idx in range(self.sample_size):
            column=negative_samples[:,idx]
            negative_column_score=self.embed_dot_layer[1+idx].forward(column,h)
            loss+=self.loss_layer[1+idx].forward(negative_column_score,negative_label)

        return loss


    def backward(self, dout=1):
        # 虽然正例和负例使用的是不同的目标词，但是它们共用了同一个隐藏向量h
        # 因此，反向传播需要同时完成两件事情：
        # - 更新正例、负例对应的词向量权重
        # - 把正例、负例产生的梯度全部汇总，传回共同的隐藏向量h
        dh = 0
        for l0, l1 in zip(self.loss_layer, self.embed_dot_layer):
            dscore = l0.backward(dout)   # SigmoidWithLoss 反传
            dh += l1.backward(dscore)    # EmbeddingDot 反传，累加 h 的梯度
        return dh
