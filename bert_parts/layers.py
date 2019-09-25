# -*- coding: utf-8 -*-
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """多头注意力"""
    def __init__(self,
                 heads,
                 head_size,
                 key_size=None,
                 initializer='glorot_uniform',
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = heads * head_size
        self.key_size = key_size if key_size else head_size
        self.kernel_initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = tf.keras.layers.Dense(units=self.heads*self.key_size, kernel_initializer=self.kernel_initializer)
        self.k_dense = tf.keras.layers.Dense(units=self.heads*self.key_size, kernel_initializer=self.kernel_initializer)
        self.v_dense = tf.keras.layers.Dense(units=self.heads*self.head_size, kernel_initializer=self.kernel_initializer)
        self.o_dense = tf.keras.layers.Dense(units=self.out_dim, kernel_initializer=self.kernel_initializer)

    def call(self, q, k, v, att_mask=None):
        """
        padding_mask: 一般来说att_mask等价于padding_mask。将qk的点积结果中padding位置的值变为负无穷，经过softmax之后就是0。
                        主要是为了屏蔽padding信息。
                      在bert和transformer里都是有这一步的。
        att_mask: 总体来说是根据应用来定制的，比如transformer做s2s时会有look_ahead_mask，防止decoder看到当前字符的下文信息。
        """
        # 先按需计算mask，这里是可以个人定制的
        if att_mask is not None:
            mask = tf.cast(tf.expand_dims(att_mask, axis=1), 'float32')  # [B, 1, T]
            ones = tf.expand_dims(tf.ones(shape=tf.shape(att_mask), dtype=tf.float32), axis=-1)  # [B, F, 1] 默认F和T是相等的
            att_mask = ones * mask

        # 输入的x值通过三个不同的dense之后得到q, k, v
        q = self.q_dense(q)
        k = self.k_dense(k)
        v = self.v_dense(v)

        """
        b: batch_size, n: num of heads, l: seq_length,  
        k: key_size, h: head_size (不额外设定时，h = k)
        """
        # reshape, [b, l, n*h] → [b, l, n, h]
        q = tf.reshape(q, [-1, tf.shape(q)[1], self.heads, self.key_size])
        k = tf.reshape(k, [-1, tf.shape(k)[1], self.heads, self.key_size])
        v = tf.reshape(v, [- 1, tf.shape(v)[1], self.heads, self.head_size])

        # attention：qk点积、padding_mask与att_mask、softmax
        a = tf.einsum('binh,bjnh->bnij', q, k) / self.key_size**0.5
        if att_mask is not None:
            att_mask = tf.expand_dims(att_mask, axis=1)  # [B, 1, F, T]
            adder = (1.0 - att_mask) * -99999
            a = a + adder
        a = tf.keras.backend.softmax(a)

        # softmax值与v做加权平均，输出形状设置成和输入类似
        o = tf.einsum('bnij,bjnh->binh', a, v)
        o = tf.reshape(o, (-1, tf.shape(o)[1], self.out_dim))
        o = self.o_dense(o)
        return o


class TransformerBlock(tf.keras.layers.Layer):
    """
    transformer
    """
    def __init__(self,
                 heads,
                 head_size,
                 intermediate_size,
                 initializer='glorot_uniform',
                 **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        assert type(head_size) == int, 'head_size type should be int'
        self.intermediate_size = intermediate_size  # feed-forward的隐层维度

        self.kernel_initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)
        self.multi_att_layer = MultiHeadAttention(heads=self.heads, head_size=self.head_size)
        self.add = tf.keras.layers.Add()
        self.ln = tf.keras.layers.LayerNormalization()
        self.ff_dense1 = tf.keras.layers.Dense(units=self.intermediate_size,
                                               activation='relu',
                                               kernel_initializer=self.kernel_initializer)
        self.ff_dense2 = tf.keras.layers.Dense(units=self.heads*self.head_size,
                                               activation='relu',
                                               kernel_initializer=self.kernel_initializer)

    def call(self, inputs, att_mask=None):
        # 多头注意力
        x0 = self.multi_att_layer(inputs, inputs, inputs, att_mask=att_mask)
        # 残差连接
        x0 = self.add([x0, inputs])
        # layer normalization
        x0 = self.ln(x0)
        # feed-forward 实际为两层dense，第二层dense输出维度可以设置，但一般和前面的输出一样，就取了n*h
        x1 = self.ff_dense1(x0)
        x1 = self.ff_dense2(x1)
        # 残差连接
        x1 = self.add([x1, x0])
        # layer normalization
        x1 = self.ln(x1)
        return x1


class SegmentEmbedding(tf.keras.layers.Layer):
    """
    表示断句信息的编码
    就第一个句子全是0，第二个句子全是1,依此类推
    输入只有一个句子的话，加不加segment embedding都是一样的
    """
    def __init__(self,
                 **kwargs):
        super(SegmentEmbedding, self).__init__(**kwargs)

    def call(self, inputs):
        """有需要的话可以自定义编写一下"""
        return inputs


class PositionEmbedding(tf.keras.layers.Layer):
    """
    位置编码
    bert的位置编码是直接加一个矩阵做embedding[max_seq_len * emb_len]，然后和输入相加。
    transformer的位置编码是根据公式计算的，计算完后和其他embedding相加
    除这两种外，还有相对位置编码，transformXL和XLNet里有用到
    """
    def __init__(self,
                 embedding_size,
                 **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.embedding_size = embedding_size

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.posembed = self.add_weight(name='pos_embedding',
                                        shape=(input_shape[1], self.embedding_size),
                                        initializer=tf.keras.initializers.get('zeros'))

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        pos_embedding = self.posembed[:seq_len]
        pos_embedding = tf.keras.backend.expand_dims(pos_embedding, 0)
        pos_embedding = tf.keras.backend.tile(pos_embedding, [batch_size, 1, 1])
        return pos_embedding + inputs


class TokenEmbedding(tf.keras.layers.Layer):
    """
    token embedding
    """
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 **kwargs):
        super(TokenEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        super(TokenEmbedding, self).build(input_shape)
        self.tokenembed = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                    output_dim=self.embedding_size)

    def call(self, inputs):
        output = self.tokenembed(inputs)
        return output


class TransTokenEmbedding(tf.keras.layers.Layer):
    """
    使用token embedding的转置矩阵使bert的输出变回到token id
    """
    def __init__(self,
                 token_embedding_weights,
                 **kwargs):
        super(TransTokenEmbedding, self).__init__(**kwargs)
        self.token_embedding_weights = token_embedding_weights

    def build(self, input_shape):
        super(TransTokenEmbedding, self).build(input_shape)
        self.transpose_token_emb = tf.keras.backend.transpose(self.token_embedding_weights)
        self.units = tf.shape(self.transpose_token_emb)[1]   # vocab数
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros')

    def call(self, inputs):
        outputs = tf.keras.backend.dot(inputs, self.transpose_token_emb)
        outputs = tf.keras.backend.bias_add(outputs, self.bias)
        outputs = tf.keras.activations.softmax(outputs)
        return outputs


class BertLayer(tf.keras.layers.Layer):
    """
    bert_parts layer
    可以用过修改transformer的层数，还有输入的最大长度，来调节模型的大小
    可以设置segment embedding，对有断句的输入做embedding
    可以设置attention mask，在特定任务中可能会对attention做特定的mask
    """
    def __init__(self,
                 vocab_size,  # 词表大小
                 embedding_size,  # embedding后的维度，后续操作基本也保持维度到这个大小
                 num_transformer_layers,  # transformer的层数
                 num_attention_heads,  # 多头注意力机制里注意力的头数
                 intermediate_size,  # transformer中feedforward里dense的units数
                 initializer_range=0.02,  # 权重初始化方差，默认0.02
                 **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        assert (embedding_size % num_attention_heads == 0), 'embed_size should be divided exactly by num_att_heads'
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)

    def build(self, input_shape):
        super(BertLayer, self).build(input_shape)
        self.tokenembed = TokenEmbedding(vocab_size=self.vocab_size,
                                         embedding_size=self.embedding_size)
        self.posembed = PositionEmbedding(embedding_size=self.embedding_size)
        self.ln = tf.keras.layers.LayerNormalization()
        self.trans_block = TransformerBlock(heads=self.num_attention_heads,
                                            head_size=int(self.embedding_size/self.num_attention_heads),
                                            intermediate_size=self.intermediate_size)

    def call(self, inputs):
        # 构建padding mask
        padding_mask = tf.keras.backend.cast(tf.keras.backend.greater(inputs, 0), 'float32')
        # token embedding
        token_embedding = self.tokenembed(inputs)
        # position embedding
        position_embedding = self.posembed(token_embedding)
        # layer norm
        x = self.ln(position_embedding)
        # transformer blocks
        # 这里是经过同一个transformer多次，参照albert
        for i in range(self.num_transformer_layers):
            x = self.trans_block(x, att_mask=padding_mask)
        return x


