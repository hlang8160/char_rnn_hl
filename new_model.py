import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq


class new_model(object):
    """docstring for new_model"""

    def __init__(self, args, training=True):
        self.args = args

        if not training:
            args.batch_size = 1
            args.seq_length = 1

        elif args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'gru':
            cell_fn == rnn.GRUCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception('model type not support')

        # 构造隐藏层
        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.nums_size)
            if training and (args.input_keep_prob < 1.0 or args.output_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        # 构造输入层
        # 占位符

        self.input_data = tf.placeholder(
            tf.int32, shape=[args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, shape=[args.batch_size, args.seq_length])
        self.intial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable(
                'softmax_w', shape=[args.nums_size, args.vocab_size])
            softmax_b = tf.get_variable('softmax_b', shape=[args.vocab_size])

        embeddings = tf.get_variable(
            'embedding', [args.vocab_size, args.nums_size])
        inputs = tf.nn.embedding_lookup(embeddings, self.input_data)
        # 输出的shape为[batch_size,seq_length,num_size]

        # 训练时输入层进行dropout

        if training and args.output_keep_prob < 1.0:
            inputs = tf.nn.dropout(
                inputs, output_keep_prob=args.output_keep_prob)

        # 现在要把shape变成[batch_size,1,num_size]
        # 将第一维的seq_length,[batch_size,1,num_size]
        inputs = tf.split(inputs, args.seq_length, 1)
        # 最后变成一个[batch_size,num_size]的一个list
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # loop函数连接num_steps步的rnn_cell，将h(t-1)的输出prev做变换然后传入h(t)作为输入
        # 这里定义的loop实际在于当我们要测试运行结果，即让机器自己写文章时，我们需要对每一步
        # 的输出进行查看。如果我们是在训练中，我们并不需要这个loop函数
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(
                tf.argmax(prev, 1))  # 输出的为vocab_size中的第某个序号
            return tf.nn.embedding_lookup(embeddings, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(
            inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        # 输出的shape为[num_steps][batch_size,num_size]

        # 这里的过程可以说基本等同于PTB模型，首先通过对output的重新梳理得到一个
        # [batch_size*seq_length, rnn_size]的输出，并将之放入softmax里，并通过sequence
        # loss by example函数进行训练

        output = tf.reshape(tf.concat(output, 1), [-1, args.nums_size])
        self.logits = tf.matmul(output, softmax_w) + \
            softmax_b  # 最后输出的维度一行vocab_size列的一维List
        self.probs = tf.nn.softmax(self.logits)

        loss = legacy_seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(
            self.targets, [-1])], [tf.ones([args.batch_size * args.seq_length], dtype=tf.float32)])
        # self.logits的shape为[batch_size*seq_length,vocab_size]
        # self.targets reshape为[1,batch_size*seq_length]
        # tf.ones(batch_size*seq_length,vocab_size)
        # 最后返回的结果长度为一维列表[1,batch_size*seq_length]

        # tf.nn.seq2seq.sequence_loss_by_example(logits, targets, weights):主要说一下这三个参数的意思和用法：
        # logits是一个二维的张量，比如是a*b,那么targets就是一个一维的张量长度为a，并且targets中元素的值是不能超过b的整形，32位的整数。也即是如果b等于4，那么targets中的元素的值都要小于4。
        # weights就是一个一维的张量长度为a，并且是一个tf.float32的数。这是权重的意思。
        #
		self.cost = cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        with tf.name_scope('cost'):
			self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

		self.final_state=last_state


		# 定义训练过程学习速率,梯度

		self.lr=tf.Variable(0.0,trainable=False)#定义学习速率将其设为不可训练
		tvars=tf.trainable_variables()  #获得所有可训练变量
		grads,_=tf.clip_by_global_norm(tf.gradients(cost,tvars),args.grad_clip)
		optimizer=tf.train.AdamOptimizer(self.lr)
		self.train_op=optimizer.apply_gradients(zip(grads,tvars))
		# slef.train_op=optimizer.minimize(grads)

		tf.summary.histogram('logits',self.logits)
		tf.summary.histogram('loss',loss)
		tf.summary.scalar('train_loss',self.loss)

	def sample(self,sess,chars,vocab,num=200,prime='The',sampling_type=1):
		state=sess.run(self.cell.zero_state(1,tf.float32))
		for char in prime[:-1]:
			x=np.zeros((1,1))
			x[0,0]=vocab[char]
			feed={self.input_data:x,self.initial_state:state}
			[state]=sess.run([self.final_state],feed)

		def weighted_pick(weights):
			t=np.cumsum(weights)
			s=np.sum(weights)
			return(int(np.searchsorted(t, np.random.rand(1)*s)))

		ret=prime
		char=prime[-1]
		for n in range(num):
			x=np.zeros((1,1))
			x[0,0]=vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret构建

