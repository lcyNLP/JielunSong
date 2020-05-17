from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import time
import os
# 设置gpu内存自增长
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 使用tf.keras.utils.get_file方法从指定地址下载数据，得到原始数据本地路径
# path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
path_to_file = "./data/jay_chou.txt"

# 打开原始数据文件并读取文本内容
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')


vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

# 对字符进行数值映射，将创建两个映射表：字符映射成数字，数字映射成字符
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
# 使用字符到数字的映射表示所有文本
text_as_int = np.array([char2idx[c] for c in text])

# 设定输入序列长度
seq_length = 100
# 获得样本总数
examples_per_epoch = len(text)//seq_length
# 将数值映射后的文本转换成dataset对象方便后续处理
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# 使用dataset的batch方法按照字符长度+1划分（要留出一个向后顺移的位置）
# drop_remainder=True表示删除掉最后一批可能小于批次数量的数据
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)


def split_input_target(chunk):
    """划分输入序列和目标序列函数"""
    # 前100个字符为输入序列，第二个字符开始到最后为目标序列
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# 使用map方法调用该函数对每条序列进行划分
dataset = sequences.map(split_input_target)


# 定义批次大小为64
BATCH_SIZE = 32

# 设定缓冲区大小，以重新排列数据集
# 缓冲区越大数据混乱程度越高，所需内存也越大
BUFFER_SIZE = 10000

# 打乱数据并分批次
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# 构建模型并且训练
# 获得词汇集大小
vocab_size = len(vocab)

# 定义词嵌入维度
embedding_dim = 512

# 定义GRU的隐层节点数量
rnn_units = 1024


# 模型包括三个层：输入层即embedding层，中间层即GRU层（详情查看）输出层即全连接层
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """模型构建函数"""
    # 使用tf.keras.Sequential定义模型
    # GRU层的参数return_sequences为True说明返回结果为每个时间步的输出，而不是最后时间步的输出
    # stateful参数为True，说明将保留每个batch数据的结果状态作为下一个batch的初始化数据
    # recurrent_initializer='glorot_uniform'，说明GRU的循环核采用均匀分布的初始化方法
    # 模型最终通过全连接层返回一个所有可能字符的概率分布.
    model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
      tf.keras.layers.GRU(rnn_units,
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(1024),
      tf.keras.layers.Dense(vocab_size)
    ])
    return model


'''模型训练部分'''
# 检查点保存至的目录
checkpoint_dir = './training_checkpoints'

# 检查点的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# 创建检测点保存的回调对象
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# 构建模型
model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

# 选择优化器
optimizer = tf.keras.optimizers.Adam()


# 编写带有装饰器@tf.function的函数进行训练
@tf.function
def train_step(inp, target):
    """
    :param inp: 模型输入
    :param tatget: 输入对应的标签
    """
    # 打开梯度记录管理器
    with tf.GradientTape() as tape:
        # 使用模型进行预测
        predictions = model(inp)
        # 使用sparse_categorical_crossentropy计算平均损失
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
    # 使用梯度记录管理器求解全部参数的梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # 使用梯度和优化器更新参数
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # 返回平均损失
    return loss


# 训练轮数
EPOCHS = 1000

#进行轮数循环
for epoch in range(EPOCHS):
    # 获得开始时间
    start = time.time()
    # 初始化隐层状态
    hidden = model.reset_states()
    # 进行批次循环
    for (batch_n, (inp, target)) in enumerate(dataset):
        # 调用train_step进行训练, 获得批次循环的损失
        loss = train_step(inp, target)
        # 每100个批次打印轮数，批次和对应的损失
        if batch_n % 1000 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch+1, batch_n, loss))

    # 每5轮保存一次检测点
    if (epoch + 1) % 5000 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    # 打印轮数，当前损失，和训练耗时
    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# 保存最后的检测点
model.save_weights(checkpoint_prefix.format(epoch=epoch))


'''生成歌词部分'''
# 恢复模型结构
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# 从检测点中获得训练后的模型参数
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

def generate_text(model, start_string):
    """
    :param model: 训练后的模型
    :param start_string: 任意起始字符串
    """
    # 要生成的字符个数
    num_generate = 200

    # 将起始字符串转换为数字（向量化）
    input_eval = [char2idx[s] for s in start_string]

    # 扩展维度满足模型输入要求
    input_eval = tf.expand_dims(input_eval, 0)

    # 空列表用于存储结果
    text_generated = []

    # 设定“温度参数”，根据tf.random_categorical方法特点，
    # 温度参数能够调节该方法的输入分布中概率的差距，以便控制随机被选中的概率大小
    temperature = 1.0

    # 初始化模型参数
    model.reset_states()

    # 开始循环生成
    for i in range(num_generate):
        # 使用模型获得输出
        predictions = model(input_eval)
        # 删除批次的维度
        predictions = tf.squeeze(predictions, 0)

        # 使用“温度参数”和tf.random.categorical方法生成最终的预测字符索引
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # 将预测的输出再扩展维度作为下一次的模型输入
        input_eval = tf.expand_dims([predicted_id], 0)

        # 将该次输出映射成字符存到列表中
        text_generated.append(idx2char[predicted_id])

    # 最后将初始字符串和生成的字符进行连接
    return (start_string + ''.join(text_generated))

# 调用函数，输出效果
print(generate_text(model, start_string="我爱你"))





