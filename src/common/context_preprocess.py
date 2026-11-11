# import numpy as np

from .choose_proc import proc


def preprocess(text,form_trans=False):#创建ID列表和两个字典
    corpus=[]
    word_to_id={}
    id_to_word={}

    text=text.replace("."," .")
    text=text.replace(","," ,")#替换符号
    split_text=[str(word) for word in text.strip().split()]
    index=0
    for word in split_text:
        if word not in word_to_id:
            word_to_id[word]=index
            id_to_word[index]=word
            corpus.append(index)
            index+=1
        else:
            corpus.append(word_to_id[word])

    if not form_trans:
        return corpus,word_to_id,id_to_word#返回的是普通列表和地图
    else:
        corpus=proc.array(corpus)
        return corpus, word_to_id, id_to_word  # 返回转格式列表和地图


def create_context_target(corpus,win_size):
    #创建上下文和目标，上下文二维，目标一维
    targets=corpus[win_size:-win_size]
    contexts=[]
    for idx in range(win_size,len(corpus)-win_size):
        context=[]
        for add in range(-win_size,win_size+1):
            if add==0:
                continue
            context.append(corpus[idx+add])
        contexts.append(context)

    contexts=proc.array(contexts)
    targets=proc.array(targets)

    return contexts,targets


def convert_one_hot(matrix,voc_size):#将np数组转换为one_hot形式
    #voc_size==len(id_to_word)
    if matrix.ndim==1:
        one_hot=proc.zeros([matrix.shape[0],voc_size],dtype=int)
        for idx,ID in enumerate(matrix):
            one_hot[idx,ID]=1

    elif matrix.ndim==2:
        one_hot=proc.zeros([matrix.shape[0],matrix.shape[1],voc_size],dtype=int)
        for idx0,context in enumerate(matrix):
            for idx1,ID in enumerate(context):
                one_hot[idx0,idx1,ID]=1

    return one_hot


#----------------------------------------------
#综合上述函数整合为普通版word2vec预处理函数
def regular_processing(text,win_size):
    corpus, word_to_id, id_to_word=preprocess(text)
    voc_size=len(id_to_word)

    contexts,targets=create_context_target(corpus,win_size)

    one_hot_contexts=convert_one_hot(contexts,voc_size)
    one_hot_targets=convert_one_hot(targets,voc_size)

    return corpus,one_hot_contexts,one_hot_targets,voc_size,word_to_id,id_to_word






def main():
    text=str(input())
    print(regular_processing(text,1))

if __name__=="__main__":
    main()


