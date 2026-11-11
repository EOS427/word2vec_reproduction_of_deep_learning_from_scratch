import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.trainer import Trainer
from src.common.optimizer import Adam#使用adam优化器
from src.model.upgrade_model.cbow import cbow
from src.model.upgrade_model.skip_gram import skip_gram
from src.data.ptb.load_ptb import load_data

def train(model):#升级版使用ptb示例
    win_size = 5
    hid_size = 100
    batch_size = 100
    max_epoch = 10

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

    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.fit(contexts, targets, max_epoch, batch_size)
    trainer.plot()

    word_vecs = model.word_vecs#展示分布式
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])


def main():
    model=str(input("测试模型："))
    train(model)

if __name__=="__main__":
    main()
