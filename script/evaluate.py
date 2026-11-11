import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from pathlib import Path

from src.common.trainer import Trainer
from src.common.optimizer import Adam#使用adam优化器
from src.model.upgrade_model.cbow import cbow
from src.model.upgrade_model.skip_gram import skip_gram
from src.data.ptb.load_ptb import load_data
from src.common.load_config import load_tool
from src.common.util import most_similar,analogy



def train(model):#使用ptb训练获得向量

    config=load_tool("word2vec_package")#加载配置文件内容
    win_size = config["win_size"]
    hid_size = config["hid_size"]
    batch_size = config["batch_size"]
    max_epoch = config["max_epoch"]

    (corpus,
     contexts,
     targets,
     voc_size,
     word_to_id,
     id_to_word)=load_data(win_size,data_type='train')#集中返回

    while True:#选择模型
        if model=="cbow":
            model = cbow(voc_size, hid_size,win_size,corpus)
            break
        elif model=="skip_gram":
            model=skip_gram(voc_size,hid_size,win_size,corpus)
            break
        else:
            print("error:retry")
            continue

    optimizer = Adam()#默认Adam优化器
    trainer = Trainer(model, optimizer)

    trainer.fit(contexts, targets, max_epoch, batch_size)

    word_matrix=model.word_vecs
    return word_to_id,id_to_word,word_matrix#返回分布式矩阵



def main():
    matrix_path=Path(__file__).parent.parent / "cache" / "word_matrix.pkl"

    if matrix_path.exists():
        with open(matrix_path,"rb",encoding="utf-8") as file:
            data=pickle.load(file)
            word_to_id=data["word_to_id"]
            id_to_word=data["id_to_word"]
            word_matrix=data["word_matrix"]
    else:
        print("无缓存，请选择模型训练（cbow或skip_gram）")#没有参数缓存时进行训练
        model=str(input("训练模型："))

        (word_to_id,
         id_to_word,
         word_matrix)=train(model)#训练获取数据

        with open(matrix_path,"wb",encoding="utf-8") as file:
            pickle.dump({"word_to_id":word_to_id,
                         "id_to_word":id_to_word,
                         "word_matrix":word_matrix}
                        ,file)
        print("训练完成")

    while True:
        print("选择测试方法：")#两种向量评估方式
        print("most_similar")
        print("analogy")
        print("exit")
        name=str(input('输入名字:'))
        if name=="most_similar":
            query=str(input("请输入查询词："))
            most_similar(query,word_to_id,id_to_word,word_matrix,top=5)
            print("done")
            continue
        elif name=="analogy":
            print("a:b==c:?")
            a=str(input("a:"))
            b=str(input("b:"))
            c=str(input("c:"))
            analogy(a,b,c, word_to_id, id_to_word, word_matrix)
            print("done")
            continue
        elif name=="exit":
            print("退出程序")
            break

        else:
            print("error:retry")
            continue

    return None








if __name__=="__main__":
    main()
