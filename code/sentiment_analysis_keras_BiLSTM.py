import re
import jieba
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences

# 设置超参数
my_lr = 1e-2
my_test_size = 0.1
my_validation_split = 0.1
my_epochs = 40
my_batch_size = 128
my_dropout = 0.2

data_path = '../data/weibo_senti_100k.csv'
wordvector_path = '../word_vector/sgns.zhihu.bigram.bz2'
# part-0 数据获取
all_data = pd.read_csv(data_path)

# print('评论数目（总体）：%d' % all_data.shape[0])
# 评论数目（总体）：119988

# print('评论数目（正向）：%d' % all_data[all_data.label==1].shape[0])
# 评论数目（正向）：59993

# print('评论数目（负向）：%d' % all_data[all_data.label==0].shape[0])
# 评论数目（负向）：59995

# 存放所有评论的列表 reviews
reviews = []

# 存放所有标签的列表 labels
labels  = []

# 打开 csv 文件，逐行读取，存放在 reviews 和 labels 中
with open(data_path, 'r', encoding = 'utf-8') as f:
    line = f.readline()
    for i in range(all_data.shape[0]):
        line = f.readline()
        reviews.append(line[2:])
        labels.append(line[0])

# count = 1
# for review in reviews:
#     print(review)
#     if count == 10:
#         break
#     else:
#         count += 1

# print(len(reviews))
# 119988
# print(len(labels))
# 119988

# part-1 构建词向量

# 使用 gensim 加载已经训练好的汉语词向
pre_word_vector = KeyedVectors.load_word2vec_format(wordvector_path, binary=False)

# 用 jieba 进行中文分词，最后将每条评论转换为了词索引的列表
train_tokens = []

for review in reviews:
    # 用正则表达式去除无意义的字符
    review = re.sub("[\s+\.\!\/_,-|$%^*(+\"\')]+|[+——！，； 。？ 、~@#￥%……&*（）]+", "", review)
    # 用 jieba 进行中文分词
    cut = jieba.cut(review)
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            # 将分出来的每个词转换为词向量中的对应索引
            cut_list[i] = pre_word_vector.key_to_index[word]
        except KeyError:
            # 如果词不在词向量中，则索引标记为0
            cut_list[i] = 0
    train_tokens.append(cut_list)

# print(len(train_tokens))
# 119988
#
# print(train_tokens[0])
# print(train_tokens[1])

# [0, 0, 3, 0, 3, 2386, 1, 814, 25, 622, 233, 42, 0, 940, 0, 42, 233, 42, 233, 42]
# [0, 3095, 1, 0, 0, 1862, 1533, 209, 7510, 10085, 52, 167, 34, 4, 9045, 1845, 11, 607, 400]

# 每段评语的长度不一，需要将索引长度标准化

# 分别计算每段评语的长度
num_tokens = np.array([len(tokens) for tokens in train_tokens])

# 选取一个长度平均值，保证尽可能多的覆盖
mid_tokens = int(np.mean(num_tokens) + 2 * np.std(num_tokens))

# 计算覆盖率
rate = np.sum( num_tokens < mid_tokens ) / len(num_tokens)

print(rate)
# 0.9434193419341934

# part-2 重构词向量

# 为了节省训练时间，抽取前50000个词构建新的词向量
num_words = 50000
embedding_dim = 300

# embedding_matrix 为一个 [num_words，embedding_dim] 的矩阵，维度为 50000 * 300
embedding_matrix = np.zeros((num_words, embedding_dim))

for i in range(num_words):
    embedding_matrix[i,:] = pre_word_vector[pre_word_vector.index_to_key[i]]

embedding_matrix = embedding_matrix.astype('float32')

# embedding_matrix.shape # (50000, 300)

# part-3 填充裁剪

# 输入的 train_tokens 是一个 list ，返回的 train_pad 是一个 numpy array ，采用 pre 填充的方式
train_pad = pad_sequences(train_tokens, maxlen = mid_tokens, padding = 'pre', truncating = 'pre')

# 超出五万个词向量的词用0代替
train_pad[train_pad >= num_words] = 0

# 准备实际输出结果向量向量，前59993好评的样本设为1，后59995差评样本设为0
train_target = np.concatenate((np.ones(59993),np.zeros(59995)))

# 用sklearn分割训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(train_pad, train_target, test_size = my_test_size, random_state = 12)

# part-4  搭建我的神经网络
model = Sequential()

model.add(Embedding(num_words, embedding_dim, weights = [embedding_matrix], input_length = mid_tokens, trainable=False))

model.add(Bidirectional(LSTM(units = 32, dropout = my_dropout, return_sequences = True)))

model.add(LSTM(units = 16, dropout = my_dropout, return_sequences=False))

model.add(Dense(1, activation = 'sigmoid'))

# 优化器
model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

# 展示模型结构
model.summary()

# part-5 训练模型

# 建立一个权重的存储点，保存训练中的最好模型
path_checkpoint = './weights.hdf5'
checkpointer = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)

# 定义early stoping如果3个epoch内validation loss没有改善则停止训练
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# 自动降低learning rate
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)

# 定义callback函数
callbacks = [earlystopping, checkpointer, lr_reduction]

# 开始训练
history = model.fit(X_train, y_train, validation_split = my_validation_split, epochs=my_epochs, batch_size=my_batch_size, callbacks=callbacks)

# part-6 模型评估

result = model.evaluate(X_test, y_test, verbose=0)

print('Loss    : {0:.4}'.format(result[0]))

print('Accuracy: {0:.4%}'.format(result[1]))
