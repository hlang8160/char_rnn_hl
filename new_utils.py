import codecs
import os
import collections
from six.moves import cPickle
import numpy as np


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, 'input.txt')
        vocab_file = os.path.join(data_dir, 'vocab.pkl')
        tensor_file = os.path.join(data_dir, 'data.npy')

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print('reading text file')
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print('loading preprocessed files')
            self.load_preprocessed(vocab_file, tensor_file)

        self.create_batches()
        self.reset_batch_pointer()

        # 处理文本的过程
        # 用codecs.open读数据,codecs是先将数据encoder 再decoder的读取方式
        # 再进行collections.Counter计数,按出现的频度进行排序 counter_pair=sorted(counter.items(),key=lambda x: -x[1])
        # 统计所有出现的chars,  char,_=zip(*counter_pair)
        # 生成一个vocab dict(zip(chars,range(len(chars))))
        # 将所有的char保存 cPickle.dump(chars,'路径')
        # 将读入的所有数据vocab上对应的,序号保存,最后input的data成为一维array []

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, 'r', encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)  # 计数
        counter_pair = sorted(
            counter.items, key=lambda x: -x[1])  # 按出现的次数从大到小排序
        self.chars, _ = zip(*counter_pair)  # 提取到char按次数从大到小排序
        self.vocab_size = len(self.chars)    #self.vocab_size是对象的属性
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
    	self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
    	if self.num_batches==0:
    		assert False, 'Not enough data. Make seq_length and batch_size small' 

    	self.tensor=self.tensor[:self.num_batches*self.batch_size*self.seq_length]
    	xdata=self.tensor
    	ydata=np.copy(self.tensor)
    	ydata[:-1]=xdata[1:]
    	ydata[-1]=xdata[0]
    	self.x_batches=np.split(xdata.reshape(self.batch_size,-1),self.num_batches,1) 
    	#将输入输出变为[[],[],[],[]],[num_batches][batch_size,seq_length]
    	self.y_batches=np.split(ydata.reshape(self.batch_size,-1),self.num_batches,1)

    def next_batch(self):
    	x,y=self.x_batches[self.pointer],self.y_batches[self.pointer]
    	self.pointer+=1
    	return x,y

    def reset_batch_pointer(self):
    	self.pointer=0