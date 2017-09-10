# 构建模型基本思路

# 输入占位
# input embedding

# 设置cell


# output

# 梯度修剪

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq


class Model_thinking():
    def __init__(self, args, training=True):
        self.args = args

        args.input_data = tf.placeholder(
            tf.int32, shape=[args.batch_size, args.seq_length])
        args.targets = tf.placeholder(
            tf.int32, shape=[args.batch_size, args.seq_length])
        # args.state=tf.placeholder(tf.int32,shape=[args.batch_size])
        # cell.intial_state

        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        else:
            raise Exception('not support model')

        cells = []
        for _ in range(arg.num_layers):
            cell = cell_fn(args.num_size)
            if training and (args.input_keep_prob < 1.0 or args.output_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(
                    cell, input_keep_prob=arg.input_keep_prob, output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        with tf.device('/cpu:0'):
            embedding = tf.get_variable(
                'embedding', shape=[args.vocab_size, args.num_size])
            inputs = tf.nn.embedding_lookup(embedding, args.input_data)

        # 开始进行输出连接
        #初始化开始状态

        args.intial_state = cell.zero_state(args.batch_size, dtype=tf.float32)

        if training and args.output_keep_prob < 1.0:
            inputs = tf.nn.dropout(
                inputs, output_keep_prob=args.output_keep_prob)

        with tf.name_scope('rnnlm'):
            softmax_w = tf.get_variable(
                'softmax_w', shape=[args.num_size, args.vocab_size])
            softmax_b = tf.get_variable('softmax_b', shape=[args.vocab_size])

        # 将inputs的shape变换
        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(inpus_, [1]) for inputs_ in inputs]

        # output输出
        def loop(prev, _):
            prev = tf.matmul(prev, args.softmax_w) + args.softmax_b
            prev_symbol = tf.stop_gradient(tf.arg_max(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(
            inputs, args.intial_state, cell, loop_function=loop if not training else None)

        # output输出变换
        outputs = tf.reshape(tf.concat(outputs, 1), [-1, args.num_size])

        # loss\

        logits = tf.matmul(outputs, args.softmax_w) + args.softmax_b
        self.probs = tf.nn.softmax(logits)
        loss = legacy_seq2seq.sequence_loss_by_example([logits],
                                                       [tf.reshape(
                                                           args.targets, [-1])],
                                                       tf.ones([args.batch_size * args.seq_length], dtype=tf.float32))

        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars), args.grad_clip)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)


#  制作RNN模型的大概步骤：
 
# # 1.定义cell类型以及模型框架(假设为lstm)：
# basic_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
# cell = tf.nn.rnn_cell.MultiRNNCell([basic_cell]*number_layers)
 
# # 2.定义输入
# input_data = tf.placeholder(tf.int32, [batch_size, sequence_length])
# target = tf.placeholder(tf.int32, [batch_size, sequence_length])
  
# # 3. init zero state
# initial_state = cell.zero_state(batch_size, tf.float32)
 
# # 4. 整理输入，可以运用PTB的方法或上文介绍的方法，不过要注意
# # 你的输入是什么形状的。最后数列要以格式[sequence_length, batch_size, rnn_size]
# # 为输入才可以。
 
# # 5. 之后为按照你的应用所需的函数运用了。这里运用的是rnn_decoder, 当然，别的可以
# # 运用，比如machine translation里运用的就是embedding_attention_seq2seq
 
# # 6. 得到输出，重新编辑输出的结构后可以运用softmax，一般loss为sequence_loss_by_example
 
# # 7. 计算loss, final_state以及选用learning rate，之后用clip_by_global norm来定义gradient
# # 并运用类似于adam来optimise算法。可以运用minimize或者apply_gradients来训练




#训练思路

#for e in range(epoches):
#	1.分配学习速率
#	sess.run(tf.assign(model.lr,args.....))
#	dataloader.reset重置输入batch
#	for b in range(num_batches):
#		2.将输入,输出喂给model.feed
#		x,y=data_loader.next_batch()
#		feed={model.input_data:x,...}
#		3.将初始状态喂给feed
#		for i,(c,h) in enumerate(model.intial_state):
#			feed[c]=state[i].c
#			feed[h]=state[i].h
#		4.开始训练,获取train_loss,state,_
#		train_loss,state,_=sess.run([model.cost,model.final_state,model.train_op])
#		5.利用tensorboard summaries
# #		                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
#                       .format(e * data_loader.num_batches + b,
#                               args.num_epochs * data_loader.num_batches,
#                               e, train_loss, end - start))

#		6.保存最后的训练模型
#
#
#
#
#
#
#