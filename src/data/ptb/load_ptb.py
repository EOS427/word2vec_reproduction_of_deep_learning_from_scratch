import os
try:
    import urllib.request#网络访问标准库
    import urllib.error#URL 错误类型
except ImportError:
    raise ImportError('Use Python3!')#确定python解释器版本
import pickle#序列化保存与读取恢复复杂数据结构库

from ...common.choose_proc import proc

from ...common.context_preprocess import preprocess,create_context_target


url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
#基础下载地址，待拼接文件名
key_file = {#原始文件
    'train':'ptb.train.txt',
    'test':'ptb.test.txt',
    'valid':'ptb.valid.txt'
}
save_file = {#缓存文件
    'train':'ptb.train.npy',
    'test':'ptb.test.npy',
    'valid':'ptb.valid.npy'
}
#不写独立的文件名，将容易变化的信息集中保存，而不是在程序各处重复写死
vocab_file = 'ptb.vocab.pkl'#词典文件

dataset_dir = os.path.dirname(os.path.abspath(__file__))#当前目录路径


def _download(file_name):#下载相应文件
    file_path = dataset_dir + '/' + file_name#检查当前目录下有没有对应文件
    if os.path.exists(file_path):
        return

    print('Downloading ' + file_name + ' ... ')

    try:
        urllib.request.urlretrieve(url_base + file_name, file_path)
        #将具体网址文件下载到目标路径
    except urllib.error.URLError:#网络连接错误
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url_base + file_name, file_path)

    print('Done')


def load_vocab():#加载字典
    vocab_path = dataset_dir + '/' + vocab_file#检查词典是否存在

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            corpus, word_to_id, id_to_word = pickle.load(f)
        return corpus, word_to_id, id_to_word#若存在加载内容


    data_type = 'train'#取训练集
    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name#目标路径

    _download(file_name)#下载原始文件

    words = open(file_path).read().replace('\n', ' <eos> ')
    #eos加入空格，区别课本保证与其他词隔开
    #读取原始文件内容，设置eos，去除首尾空字符并拆分

    corpus, word_to_id, id_to_word=preprocess(words)#区别课本使用已包装函数处理数据

    with open(vocab_path, 'wb') as f:
        pickle.dump((corpus, word_to_id, id_to_word), f)#保存数据为pkl

    return corpus, word_to_id, id_to_word


#——————————————————————————————————————————————————————————————————————————
#打包以上函数，集中为一体，检查/下载数据以及预处理
def load_data(win_size,data_type='train'):#加载数据，默认加载训练集
    if data_type == 'val': data_type = 'valid'
    #valid：训练过程中使用验证集检查模型效果，防止过拟合
    save_path = dataset_dir + '/' + save_file[data_type]#缓存路径

    corpus, word_to_id, id_to_word = load_vocab()#加载字典

    voc_size = len(id_to_word)
    context,target=create_context_target(corpus,win_size)#创建上下文和目标词

    if os.path.exists(save_path):#检查缓存是否存在
        corpus = proc.load(save_path).tolist()#读取的是经转化的数组，需还原为列表
        return corpus,context,target,voc_size,word_to_id,id_to_word

    file_name = key_file[data_type]#原始文件名
    # file_path = dataset_dir + '/' + file_name#原始文件目标路径
    _download(file_name)#检查/下载原始文件

    proc.save(save_path, corpus)#numpy和cupy均保存为.npy，保存读取互相通用
    #普通列表实际上会自动转化为对应类型数组保存，上文load时需注意格式

    return corpus,context,target,voc_size,word_to_id,id_to_word


def load(win_size,data_type):#加载训练，验证，测试集
    for data_type in ('train', 'val', 'test'):
        load_data(win_size,data_type=data_type)

