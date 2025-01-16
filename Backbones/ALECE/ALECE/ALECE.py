import tensorflow as tf
print(tf.__version__)
import numpy as np
import sys
import os
sys.path.append("../")
sys.path.append("/data1/liuhaoran/ALECE/")
from src.utils import tf_utils
from src import arg_parser
from keras import backend as K

class attention(tf.keras.layers.Layer):
    def __init__(self,
                 args,
                 if_self_attn=True,trainable=True
                 ):
        super(attention, self).__init__(trainable=trainable)
        self.attn_head_key_dim = args.attn_head_key_dim
        self.num_attn_heads = args.num_attn_heads
        self.if_self_attn = if_self_attn
        self.attr_states_dim = args.n_bins
        self.query_part_feature_dim = args.query_part_feature_dim
        self.dropout_rate = 0.0
        if args.use_dropout == 1:
            self.dropout_rate = args.dropout_rate
        self.feed_forward_dim = args.feed_forward_dim
        self._dtype = tf.float32
        if args.use_float64 == 1:
            self._dtype = tf.float64

        # Multi-head self-attention.
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_attn_heads,
            key_dim=self.attn_head_key_dim,  # Size of each attention head for query Q and key K.
            dropout=self.dropout_rate, trainable=trainable
        )

        # Point-wise feed-forward network.
        if self.if_self_attn:
            self.ffn, self.regularization = tf_utils.point_wise_feed_forward_network(self.attr_states_dim, args.feed_forward_dim, self._dtype, trainable=trainable)
        else:
            self.ffn, self.regularization = tf_utils.point_wise_feed_forward_network(self.query_part_feature_dim, args.feed_forward_dim, self._dtype, trainable=trainable)


        # Layer normalization.
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, trainable=trainable)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, trainable=trainable)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate, trainable=trainable)



    def call(self, x, training):
        # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_output = self.mha(
            query=x,  # Query Q tensor.
            value=x,  # Value V tensor.
            key=x,  # Key K tensor.
            attention_mask=None,  # A boolean mask that prevents attention to certain positions.
            training=training,  # A boolean indicating whether the layer should behave in training mode.
        )
        # tf.print('attn_output.shape =', attn_output.shape)

        # Multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layernorm1(x + attn_output)  # Shape `(batch_size, input_seq_len, d_model)`
        # tf.print('out1.shape =', out1.shape)

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out1)  # Shape `(batch_size, input_seq_len, d_model)`
        ffn_output = self.dropout1(ffn_output, training=training)
        # tf.print('ffn_output.shape =', ffn_output.shape)

        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        out2 = self.layernorm2(out1 + ffn_output)  # Shape `(batch_size, input_seq_len, d_model)`.

        return out2

    def self_attn(self, x, training):
        # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_output = self.mha(
            query=x,  # Query Q tensor.
            value=x,  # Value V tensor.
            key=x,  # Key K tensor.
            attention_mask=None,  # A boolean mask that prevents attention to certain positions.
            training=training,  # A boolean indicating whether the layer should behave in training mode.
        )
        # tf.print('attn_output.shape =', attn_output.shape)

        # Multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layernorm1(x + attn_output)  # Shape `(batch_size, input_seq_len, d_model)`
        # tf.print('out1.shape =', out1.shape)

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out1)  # Shape `(batch_size, input_seq_len, d_model)`
        ffn_output = self.dropout1(ffn_output, training=training)
        # tf.print('ffn_output.shape =', ffn_output.shape)

        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        out2 = self.layernorm2(out1 + ffn_output)  # Shape `(batch_size, input_seq_len, d_model)`.

        return out2

    def attn(self, x, q, training):
        # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_output = self.mha(
            query=q,  # Query Q tensor.
            value=x,  # Value V tensor.
            key=x,  # Key K tensor.
            attention_mask=None,  # A boolean mask that prevents attention to certain positions.
            training=training  # A boolean indicating whether the layer should behave in training mode.
        )
        # tf.print('attn_output.shape =', attn_output.shape)

        # Multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layernorm1(q + attn_output)  # Shape `(batch_size, input_seq_len, d_model)`
        # tf.print('out1.shape =', out1.shape)

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out1)  # Shape `(batch_size, input_seq_len, d_model)`
        ffn_output = self.dropout1(ffn_output, training=training)
        # tf.print('ffn_output.shape =', ffn_output.shape)

        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        out2 = self.layernorm2(out1 + ffn_output)  # Shape `(batch_size, input_seq_len, d_model)`.

        return out2


class attnModel_mask(tf.keras.Model):
    """
    encoder 相同
    decoder 加入mask
    """
    def __init__(self,
                 args
                 ):
        super(attnModel_mask, self).__init__()

        self.num_self_attn_layers = args.num_self_attn_layers
        self.num_cross_attn_layers = args.num_cross_attn_layers
        self.num_attrs = args.num_attrs
        self.attr_states_dim = args.n_bins
        self._dtype = tf.float32
        if args.use_float64 == 1:
            self._dtype = tf.float64

        self.use_positional_embedding = (args.use_positional_embedding != 0)
        self.use_dropout = (args.use_dropout != 0)

        # tf.print('self.pos_encoding.shape =', self.pos_encoding.shape)
        self.regularization = tf.constant(0, dtype=self._dtype)
        self.mask_regularization = tf.constant(0, dtype=self._dtype)

        # Data-encoder
        self.self_attn_layers = [
            attention(
                args,
                if_self_attn=True
            )
            for _ in range(self.num_self_attn_layers)]

        # Query-analyzer
        self.cross_attn_layers = [
            attention(
                args,
                if_self_attn=False
            )
            for _ in range(self.num_cross_attn_layers)]

        # column-mask 跟cross层数一样
        self.attn_mask_layers = [
            attention(
                args,
                if_self_attn=False
            )
            for _ in range(self.num_cross_attn_layers)]


        for layer in self.self_attn_layers:
            self.regularization += layer.regularization

        for layer in self.cross_attn_layers:
            self.regularization += layer.regularization

        for layer in self.attn_mask_layers:
            self.mask_regularization += layer.regularization

        mlp_layers = []
        for _ in range(args.mlp_num_layers - 1):
            layer = tf.keras.layers.Dense(args.mlp_hidden_dim,
                                          activation='elu',
                                          dtype=self._dtype,
                                          kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                                          bias_regularizer=tf.keras.regularizers.L2(1e-4))
            mlp_layers.append(layer)
            self.regularization += tf.math.reduce_sum(layer.losses)

        final_layer = tf.keras.layers.Dense(1,
                                            dtype=self._dtype,
                                            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                                            bias_regularizer=tf.keras.regularizers.L2(1e-4))

        self.regularization += tf.math.reduce_sum(final_layer.losses)
        mlp_layers.append(final_layer)
        self.reg = tf.keras.Sequential(mlp_layers)

        # Dropout.
        self.dropout_rate = 0.0
        if args.use_dropout:
            self.dropout_rate = args.dropout_rate
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # self.x_normalize = tf.keras.layers.Normalization()
        # self.q_normalize = tf.keras.layers.Normalization()

    def call(self, x, q, training):
        """
        :param c: Shape `(batch_size, num_attrs, attr_states_dim)
        :param x: Shape `(batch_size, num_attrs, attr_states_dim)
        :param q: Shape `(batch_size, 1, query_part_features_dim)
        :param training: tf tensor
        :return:
        """
        # with self.strategy.scope():
        seq_len = tf.shape(x)[1]

        if self.use_positional_embedding:
            x *= tf.math.sqrt(tf.cast(self.attr_states_dim, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]
        # Add dropout.
        if self.use_dropout:
            x = self.dropout(x, training=training)

        # Data-encoder forward
        m_rec = []

        ### encoding True
        for i in range(self.num_self_attn_layers):
            x = self.self_attn_layers[i].attn(x, x, training)
            
            # m = self.attn_mask_layers[i].attn(x, x, training)
            # x = self.self_attn_layers[i].attn(x, x, training) ##
            # ones_like_q = tf.ones_like(x)  
            # mu = tf.multiply(ones_like_q, 1/self.num_attrs) 
            # x1 = tf.multiply(x, tf.subtract(tf.constant(1, dtype=tf.float32), m))
            # x2 = tf.multiply(mu, m)
            # x = tf.nn.softmax(tf.add(x1, x2))
            # m_rec.append(m)
            ### 

        # Query-analyzer forward
        ### decoding
        for i in range(self.num_cross_attn_layers):
            m = self.attn_mask_layers[i].attn(x, q, training)
            q = self.cross_attn_layers[i].attn(x, q, training) ##
            ones_like_q = tf.ones_like(q)  
            mu = tf.multiply(ones_like_q, 1/self.num_attrs) 
            x1 = tf.multiply(q, tf.subtract(tf.constant(1, dtype=tf.float32), m))
            x2 = tf.multiply(mu, m)
            x = tf.nn.softmax(tf.add(x1, x2))
            m_rec.append(m)
            ### 

        preds = self.reg(q, training=training)

        return preds, tf.convert_to_tensor(m_rec)

    def call_b(self, x, q, training):
        """
        :param c: Shape `(batch_size, num_attrs, attr_states_dim)
        :param x: Shape `(batch_size, num_attrs, attr_states_dim)
        :param q: Shape `(batch_size, 1, query_part_features_dim)
        :param training: tf tensor
        :return:
        """
        # with self.strategy.scope():
        seq_len = tf.shape(x)[1]

        if self.use_positional_embedding:
            x *= tf.math.sqrt(tf.cast(self.attr_states_dim, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]
        # Add dropout.
        if self.use_dropout:
            x = self.dropout(x, training=training)

        # Data-encoder forward
        m_rec = []

        for i in range(self.num_self_attn_layers):
            x = self.self_attn_layers[i].attn(x, x, training)
            m = self.attn_mask_layers[i].attn(x, x, training)
            # !!!不知道常量写多少，先写个1/n试试，n是num_self_attn_layers
            ones_like_x = tf.ones_like(x)  
            x = tf.subtract(x, tf.multiply(m, tf.multiply(ones_like_x, 1/self.num_self_attn_layers)  ))
            x = tf.sigmoid(x)
            # 蒽或许可以这样存吗
            m_rec.append(m)


        # Query-analyzer forward
        for i in range(self.num_cross_attn_layers):
            m = self.attn_mask_layers[i].attn(x, q, training)
            # q = self.cross_attn_layers[i].attn(x, q, training)
            # # m = self.attn_mask_layers[i].attn(x, q, training)
            # # !!!不知道常量写多少，先写个1/n试试，n是num_self_attn_layers
            # ones_like_x = tf.ones_like(q)  
            # a = tf.multiply(ones_like_x, 1/self.num_self_attn_layers) 
            # q = tf.subtract(q, tf.multiply(m, tf.multiply(ones_like_x, 1/self.num_self_attn_layers)  ))
            # q = tf.sigmoid(q)
            # # 蒽或许可以这样存吗
            # m_rec.append(m)

        preds = self.reg(q, training=training)

        return preds, tf.convert_to_tensor(m_rec)

class attnModel(tf.keras.Model):
    def __init__(self,
                 args, trainable=True
                 ):
        super(attnModel, self).__init__()

        self.num_self_attn_layers = args.num_self_attn_layers
        self.num_cross_attn_layers = args.num_cross_attn_layers
        self.num_attrs = args.num_attrs
        self.attr_states_dim = args.n_bins
        self._dtype = tf.float32
        if args.use_float64 == 1:
            self._dtype = tf.float64

        self.use_positional_embedding = (args.use_positional_embedding != 0)
        self.use_dropout = (args.use_dropout != 0)

        # tf.print('self.pos_encoding.shape =', self.pos_encoding.shape)
        self.regularization = tf.constant(0, dtype=self._dtype)

        # Data-encoder
        self.self_attn_layers = [
            attention(
                args,
                if_self_attn=True, trainable=trainable
            )
            for _ in range(self.num_self_attn_layers)]

        # Query-analyzer
        self.cross_attn_layers = [
            attention(
                args,
                if_self_attn=False, trainable=trainable
            )
            for _ in range(self.num_cross_attn_layers)]

        for layer in self.self_attn_layers:
            self.regularization += layer.regularization

        for layer in self.cross_attn_layers:
            self.regularization += layer.regularization

        mlp_layers = []
        for _ in range(args.mlp_num_layers - 1):
            layer = tf.keras.layers.Dense(args.mlp_hidden_dim,
                                          activation='elu',
                                          dtype=self._dtype,
                                          kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                                          bias_regularizer=tf.keras.regularizers.L2(1e-4), trainable=trainable)
            mlp_layers.append(layer)
            self.regularization += tf.math.reduce_sum(layer.losses)

        final_layer = tf.keras.layers.Dense(1,
                                            dtype=self._dtype, 
                                            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                                            bias_regularizer=tf.keras.regularizers.L2(1e-4), trainable=trainable)

        self.regularization += tf.math.reduce_sum(final_layer.losses)
        mlp_layers.append(final_layer)
        self.reg = tf.keras.Sequential(mlp_layers)

        # Dropout.
        self.dropout_rate = 0.0
        if args.use_dropout:
            self.dropout_rate = args.dropout_rate
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, trainable=trainable)

        # self.x_normalize = tf.keras.layers.Normalization()
        # self.q_normalize = tf.keras.layers.Normalization()



    def call(self, x, q, training):
        """
        :param x: Shape `(batch_size, num_attrs, attr_states_dim)
        :param q: Shape `(batch_size, 1, query_part_features_dim)
        :param training: tf tensor
        :return:
        """
        # with self.strategy.scope():
        seq_len = tf.shape(x)[1]

        if self.use_positional_embedding:
            x *= tf.math.sqrt(tf.cast(self.attr_states_dim, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]
        # Add dropout.
        if self.use_dropout:
            x = self.dropout(x, training=training)

        # Data-encoder forward
        for i in range(self.num_self_attn_layers):
            x = self.self_attn_layers[i].attn(x, x, training)

        # Query-analyzer forward
        for i in range(self.num_cross_attn_layers):
            q = self.cross_attn_layers[i].attn(x, q, training)

        preds = self.reg(q, training=training)

        return preds,x

import random
attn_train_step_signature = [
    (tf.TensorSpec(shape=(None, None), dtype=tf.float32),
     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
     tf.TensorSpec(shape=(None), dtype=tf.float32),
     tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
     ),
]

from ot_util import OT

class ALECE(object):
    def __init__(self, args,trainable=True):
        self.model_name = 'ALECE'
        tf.debugging.set_log_device_placement(True)
        gpus = tf.config.list_logical_devices('GPU')
        self.strategy = tf.distribute.MirroredStrategy(gpus)
        self.model2 = None
        self.model3 = None
        self.model4 = None
        self.e = tf.Variable(0)
        self.a = []
        self.aT = []
        self.datas = []
        self.useD = []
        self.ot = OT('cpu')
        with self.strategy.scope():
            self.model_name = args.model
            self.num_attrs = args.num_attrs
            self.attr_states_dim = args.n_bins
            self.query_part_feature_dim = args.query_part_feature_dim
            self.attn_model = attnModel(args,trainable=trainable)  #nomaskversion
            #self.attn_model = attnModel_mask(args)
            self.mse_loss_object = tf.keras.losses.MeanSquaredError()

            self.require_init_train_step = False
            self.lr = args.lr


    def ckpt_init(self, ckpt_dir):
        with self.strategy.scope():
            self.ckpt_step = tf.Variable(-1, trainable=False)
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
            self.ckpt_dir = ckpt_dir

            self.ckpt = tf.train.Checkpoint(
                step=self.ckpt_step,
                model=self.attn_model,
                # x_normalize=self.x_normalize,
                # q_normalize=self.q_normalize,
                optimizer=self.optimizer
            )
            self.manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=3)

    def ckpt_reinit(self, ckpt_dir):
        with self.strategy.scope():
            self.ckpt_step.assign(-1)
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
            self.ckpt_dir = ckpt_dir

            self.ckpt = tf.train.Checkpoint(
                step=self.ckpt_step,
                model=self.attn_model,
                # x_normalize=self.x_normalize,
                # q_normalize=self.q_normalize,
                optimizer=self.optimizer
            )
            self.manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=3)


    def set_model_name(self, mname):
        self.model_name = mname

    def save(self, ckpt_step):
        self.ckpt_step.assign(ckpt_step)
        self.manager.save()

    def restore(self):
        if os.path.exists(self.ckpt_dir):
            # vars = tf.train.list_variables(self.ckpt_dir)
            self.ckpt.restore(self.manager.latest_checkpoint)
        return self.ckpt_step

    def compile(self, train_data):
        # train_X, train_Q, train_labels = train_data
        pass

    def forward(self, X, Q, training):
        with self.strategy.scope():

            X = tf.reshape(X, [-1, self.num_attrs, self.attr_states_dim])
            Q = tf.reshape(Q, [-1, 1, self.query_part_feature_dim])
            # Q = tf.reshape(Q, [-1, 1, Q.shape[1]])
            preds = self.attn_model(
                x=X,
                q=Q,
                training=training
            )
            #return preds,1 #nomaskversion
            return preds


    def build_loss_wzj(self, train_X, train_Q, train_weights, train_labels):
        # preds = self.forward(train_X, train_Q, True)
        # loss = self.mse_loss_object(train_labels, preds, sample_weight=train_weights) + self.attn_model.regularization
        preds, m_rec = self.forward(train_X, train_Q, True)
        loss = self.mse_loss_object(train_labels, preds, sample_weight=train_weights) + self.attn_model.regularization
        loss_m = self.mse_loss_object(m_rec, tf.zeros_like(m_rec)) + self.attn_model.mask_regularization
        return loss+loss_m
    
    def build_loss_wzj_ot(self, train_X, train_Q, train_weights, train_labels, cur_epoch=None):
        preds, x = self.forward(train_X, train_Q, True)
        loss = self.mse_loss_object(train_labels, preds, sample_weight=train_weights) + self.attn_model.regularization
        if self.transfer and cur_epoch > self.pretrain_epoch:
            loss_ot = 0.1 * self.ot(tf.squeeze(x[0,:,:]).cpu().numpy(), source_interv)
            loss_ot = tf.tensor(loss_ot)
            loss += 0.1*loss_ot
        return loss

    def build_loss(self, train_X, train_Q, train_weights, train_labels):
        preds,_ = self.forward(train_X, train_Q, True)
        loss = self.mse_loss_object(train_labels, preds, sample_weight=train_weights) + self.attn_model.regularization
        return loss
    
    def build_loss_d(self, train_X, train_Q, train_weights, train_labels):
        
        if self.e <300:
            preds2,x1 = self.model4.forward(train_X, train_Q, False)
            preds,x2  = self.forward(train_X, train_Q, True)
            loss = self.mse_loss_object(train_labels, preds, sample_weight=train_weights) + self.attn_model.regularization + self.mse_loss_object(preds2, preds, sample_weight=train_weights)
            ## loss_ot = 0.1 * self.ot(tf.squeeze(x1[0,:,:]).cpu().numpy(), tf.squeeze(x2[0,:,:]).cpu().numpy())
            # loss_ot = 0.1 * self.ot(tf.squeeze(x1[0, :, :]), tf.squeeze(x2[0, :, :])) true
            # loss += 0.1*loss_ot
        elif self.e < 600:
            preds2,x1 = self.model3.forward(train_X, train_Q, False)
            preds,x2  = self.forward(train_X, train_Q, True)
            loss = self.mse_loss_object(train_labels, preds, sample_weight=train_weights) + self.attn_model.regularization + self.mse_loss_object(preds2, preds, sample_weight=train_weights)
            ## loss_ot = 0.1 * self.ot(tf.squeeze(x1[0,:,:]).cpu().numpy(), tf.squeeze(x2[0,:,:]).cpu().numpy())
            # loss_ot = 0.1 * self.ot(tf.squeeze(x1[0, :, :]), tf.squeeze(x2[0, :, :])) true
            # loss += 0.1*loss_ot
        elif self.e < 900:
            preds2,x1 = self.model2.forward(train_X, train_Q, False)
            preds,x2  = self.forward(train_X, train_Q, True)
            loss = self.mse_loss_object(train_labels, preds, sample_weight=train_weights) + self.attn_model.regularization + self.mse_loss_object(preds2, preds, sample_weight=train_weights)
            ## loss_ot = 0.1 * self.ot(tf.squeeze(x1[0,:,:]).cpu().numpy(), tf.squeeze(x2[0,:,:]).cpu().numpy())
            # loss_ot = 0.1 * self.ot(tf.squeeze(x1[0, :, :]), tf.squeeze(x2[0, :, :]))
            # loss += 0.1*loss_ot
        else:
            preds,x1 = self.forward(train_X, train_Q, True)
            loss = self.mse_loss_object(train_labels, preds, sample_weight=train_weights) + self.attn_model.regularization 
        return loss
    
    def build_loss_d_ot(self, train_X, train_Q, train_weights, train_labels):
        
        if self.e < 900:
            preds2,x1 = self.model4.forward(train_X, train_Q, False)
            preds,x2  = self.forward(train_X, train_Q, True)
            loss = self.mse_loss_object(train_labels, preds, sample_weight=train_weights) + self.attn_model.regularization 
            #+ self.mse_loss_object(preds2, preds, sample_weight=train_weights)
            # loss_ot = 0.1 * self.ot(tf.squeeze(x1[0,:,:]).cpu().numpy(), tf.squeeze(x2[0,:,:]).cpu().numpy())
            loss_ot = 0.1 * self.ot(tf.squeeze(x1[0, :, :]), tf.squeeze(x2[0, :, :]))
            loss += 0.1*loss_ot
        else:
            preds,x1 = self.forward(train_X, train_Q, True)
            loss = self.mse_loss_object(train_labels, preds, sample_weight=train_weights) + self.attn_model.regularization 
        return loss
    
    @tf.function
    def f(self,x, y,sample_weight):
        return self.mse_loss_object(x, y, sample_weight=sample_weight)
    
    
    @tf.function
    def build_loss_dd(self,train_data):
        (train_X, train_Q, train_weights, train_labels) = train_data
        #  处理： 600- 900 记录分别使用上一阶段loss表现差的样本，50%
        flag = False
        a = 0.0
        aT = 0.0
        if self.e <300:
            preds2 = self.model4.forward(train_X, train_Q, False)
            preds = self.forward(train_X, train_Q, True)
            a =  self.f(train_labels, preds, sample_weight=train_weights)
            aT = self.f(preds2, preds, sample_weight=train_weights)
            loss = a + self.attn_model.regularization + aT   
        elif self.e < 600:
            preds2 = self.model3.forward(train_X, train_Q, False)
            preds = self.forward(train_X, train_Q, True)
            a =  self.f(train_labels, preds, sample_weight=train_weights)
            aT = self.f(preds2, preds, sample_weight=train_weights)
            loss = a + self.attn_model.regularization + aT     
        elif self.e < 900:
            preds2 = self.model2.forward(train_X, train_Q, False)
            preds = self.forward(train_X, train_Q, True)
            a =  self.f(train_labels, preds, sample_weight=train_weights)
            aT = self.f(preds2, preds, sample_weight=train_weights)
            loss = a + self.attn_model.regularization + aT
        else:
            preds = self.forward(train_X, train_Q, True)
            loss = self.f(train_labels, preds, sample_weight=train_weights)
    
        # 
        return loss
    
    @tf.function  
    def print_e(self,e):  
        print("Value of e:", e)  
        return e
    
    ## 真正调用
    @tf.function(input_signature=attn_train_step_signature)
    def train_step(self, train_data):
        (train_X, train_Q, train_weights, train_labels) = train_data
        with tf.GradientTape() as tape:
            loss = self.build_loss(train_X, train_Q, train_weights, train_labels)
        gradients = tape.gradient(loss, self.attn_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.attn_model.trainable_variables))




    def eval_validation(self, valid_data):
        (valid_X, valid_Q,valid_labels) = valid_data
        preds,_ = self.forward(valid_X, valid_Q, training=False)
        loss = self.mse_loss_object(valid_labels, preds)
        return loss
    
    # data_re == True 调用
    def eval_validation2(self, valid_data):
        (valid_X, valid_Q, ee,valid_labels) = valid_data
        preds = self.forward(valid_X, valid_Q, training=False)
        loss = self.mse_loss_object(valid_labels, preds)
        return loss

    # data_re == True 调用
    def eval_validation2_T(self, valid_data):
        (valid_X, valid_Q, ee,valid_labels) = valid_data
        preds = None
        if self.e <300:
            preds = self.model4.forward(valid_X, valid_Q, training=False)
        elif self.e < 600:
            preds = self.model3.forward(valid_X, valid_Q, training=False)
        elif self.e < 900:
            preds = self.model2.forward(valid_X, valid_Q, training=False)
        loss = self.mse_loss_object(valid_labels, preds)
        return loss

    def eval_test(self, test_data):
        (test_X, test_Q, _) = test_data
        preds,_ = self.forward(test_X, test_Q, training=False)
        return preds
