import numpy
import time
import matplotlib.pyplot as plt
from .choose_proc import proc
from .util import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None#打印频率
        self.current_epoch = 0#已完成训练数

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):#默认不进行梯度裁剪
        #x:contexts,t:targets
        data_size = len(x)#计算参与计算上下文数量
        max_iters = data_size // batch_size#计算迭代次数，地板除忽略多余数据
        self.eval_interval = eval_interval#记录损失的设定间隔
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()#开始时间戳
        for epoch in range(max_epoch):#训练轮数控制
            # 打乱
            idx = numpy.random.permutation(numpy.arange(data_size))
            #创建0-len(x)-1的np数组，随机打乱
            #每次训练前打乱次序
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]
                #每次迭代从左往右取一批数据

                # 计算梯度，更新参数
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)#对于重复使用的权重，将共享的权重整合为1个

                if max_grad is not None:#梯度裁剪，避免梯度过大，保持训练稳定
                    clip_grads(grads, max_grad)

                optimizer.update(params, grads)#更新数据
                total_loss += loss#累加损失，用于计算平均值
                loss_count += 1#损失计算次数

                # 评价
                if (eval_interval is not None) and (iters % eval_interval) == 0:#迭代至记录周期
                    avg_loss = total_loss / loss_count#平均损失
                    elapsed_time = time.time() - start_time#计算开始到现在累计时间
                    print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))#损失列表添加平均损失
                    total_loss, loss_count = 0, 0#每周期归零

            self.current_epoch += 1#训练数+1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))#注意plt要求使用np数组作为参数
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()


def remove_duplicate(params, grads):
    #将参数列表中重复的权重整合为1个，
    #加上与该权重对应的梯度

    params, grads = params[:], grads[:]  #copy列表

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):

                # 在共享权重的情况下
                if params[i] is params[j]:#是否为相同对象
                    grads[i] += grads[j]  # 加上梯度
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)#同时剔除重复参数和梯度

                # 在作为转置矩阵共享权重的情况下（weight tying）
                elif (params[i].ndim == 2
                      and params[j].ndim == 2
                      and params[i].T.shape == params[j].shape
                      and proc.all(params[i].T == params[j])):#对于两个二维参数矩阵通过转置关系共享权重的情况

                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads
