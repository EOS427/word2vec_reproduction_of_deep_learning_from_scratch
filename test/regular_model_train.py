import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.trainer import Trainer
from src.common.optimizer import Adam#使用adam优化器
from src.model.regular_model.cbow import cbow
from src.model.regular_model.skip_gram import skip_gram
from src.common.context_preprocess import regular_processing

def train(model):
    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    text = 'You say goodbye and I say hello.'

    (corpus,
     one_hot_contexts,
     one_hot_targets,
     vocab_size,
     word_to_id,
     id_to_word)=regular_processing(text,window_size)#集中返回

    while True:#选择模型
        if model=="cbow":
            model = cbow(vocab_size, hidden_size)
            break
        elif model=="skip_gram":
            model=skip_gram(vocab_size,hidden_size)
            break
        else:
            print("error:retry")
            continue

    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.fit(one_hot_contexts, one_hot_targets, max_epoch, batch_size)
    trainer.plot()

    word_vecs = model.word_vecs#展示分布式
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])


def main():
    model=str(input("测试模型："))
    train(model)

if __name__=="__main__":
    main()
