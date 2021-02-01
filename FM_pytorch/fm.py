import torch
import numpy as np

class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        #这里filed_dims以电影的数据为例子，x有两列user_id,movie_id,这里的field_dims=[610，193609]是两个id的最大值
        #根据其最大值共同构建统一的embedding字典，也就是字典共存放610+193609个向量
        self.embedding = torch.nn.Embedding(sum(field_dims), output_dim)
        #线性回归的偏置项
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        #这里的offsets表示偏移项，因为user_id和movie_id共同用一个embedding字典，所以要通过偏移来找各个id对应得embedding向量
        #这里0~609是表示的user_id的向量，而610~610+193609表示的是movie_id的向量，这样即便有多个特征也能根据此方法共享一个embedding字典
        #所以offsets=[0,610]，第1特征的偏移量肯定是0了，后面依次根据前面的累加
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        #这里加上偏移项才能从embedding字典取到正确的向量
        #因为x.shape=[batch_size, num_fields],offsets=[2],所以通过unsqueeze(0)变成offsets=[1,2],这样shape差不多才能和x相加
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        #dim=1，表示各个特征的embedding一行相加最后+bias得到单个样本的线性回归结果
        return torch.sum(self.embedding(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class  FeatureInteraction(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.interaction = FeatureInteraction(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        #线性回归部分+特征交叉部分
        x = self.linear(x) + self.interaction(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
