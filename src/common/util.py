import os
import sys

from .choose_proc import proc

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    #不归一化使用余弦函数计算相似度，取前top个相似值，归一形式见下文
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = proc.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

def cos_similarity(x, y, eps=1e-8):#计算余弦相似度
    nx = x / (proc.sqrt(proc.sum(x ** 2)) + eps)
    ny = y / (proc.sqrt(proc.sum(y ** 2)) + eps)
    return proc.dot(nx, ny)



def clip_grads(grads, max_norm):#全局范数裁剪
    #计算所有梯度拼在一起后的总长度，也就是全局 L2 范数；如果这个总长度超过 max_norm，就把所有梯度按同一个比例缩小。
    total_norm = 0
    for grad in grads:
        total_norm += proc.sum(grad ** 2)
    total_norm = proc.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate



def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):#检查词是否存在于词典中
        if word not in word_to_id:
            print('%s is not found' % word)
            return

    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = (word_matrix[word_to_id[a]],
                           word_matrix[word_to_id[b]],
                           word_matrix[word_to_id[c]])
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)#归一化，可能产生无效值，处理见下文

    similarity = proc.dot(word_matrix, query_vec)#点积计算相似度
    #向量矩阵和查询向量已经归一化，结果为余弦
    #np/cp一维数组在矩阵乘法中不是“永远默认 n×1”，而是：
    # 在左边参与 @ 时偏向当作行向量；
    # 在右边参与 @ 时偏向当作列向量；
    # 最后结果中的单维度会被去掉。

    if answer is not None:
        print("==>" + answer + ":" + str(proc.dot(word_matrix[word_to_id[answer]], query_vec)))
        #若需计算是否提供正确答案，则输出答案和查询两向量的点积

    count = 0
    for i in (-1 * similarity).argsort():#*-1反转排序，输出从大到小的元素对应序列
        if proc.isnan(similarity[i]):#检查是否为无效量（not_a_number）,例如零为分母
            continue
        if id_to_word[i] in (a, b, c):#排除参数自身
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))#python字符格式化，参数和字符位置一一对应

        count += 1
        if count >= top:#只取前top个相近值
            return


def normalize(x):#归一化，将向量的长度调整为1但保留向量的方向
    if x.ndim == 2:
        s = proc.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = proc.sqrt((x * x).sum())
        x /= s
    return x