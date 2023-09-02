import torch
from search_space.act_pool import ActPool
from search_space.conv_pool import ConvPool
from search_space.norm_pool import NormPool


class IdentityOperation_2(torch.nn.Module):

    def __init__(self,
                 hidden_dimension,
                 architecture):

        super(IdentityOperation_2, self).__init__()

        self.layer1_act_pool = ActPool()
        self.layer2_act_pool = ActPool()

        self.layer1_conv = ConvPool(hidden_dimension, hidden_dimension).get_conv(architecture[0])
        self.layer1_norm = NormPool(hidden_dimension).get_norm(architecture[1])
        self.layer1_act = self.layer1_act_pool.get_act(architecture[2])

        self.layer2_conv = ConvPool(hidden_dimension, hidden_dimension).get_conv(architecture[3])
        self.layer2_norm = NormPool(hidden_dimension).get_norm(architecture[4])
        self.layer2_act = self.layer2_act_pool.get_act(architecture[5])

    def forward(self, x, edge_index):

        skip_connection_x_list = []

        x = self.layer1_conv(x, edge_index)
        x = self.layer1_norm(x)
        x = self.layer1_act(x)
        skip_connection_x_list.append(x)

        x = self.layer2_conv(x, edge_index)
        x = self.layer2_norm(x)
        x = self.layer2_act(x)
        skip_connection_x_list.append(x)

        skip_connection_x = sum(skip_connection_x_list)

        return skip_connection_x


class IdentityOperation_1(torch.nn.Module):

    def __init__(self,
                 hidden_dimension,
                 architecture):

        super(IdentityOperation_1, self).__init__()

        self.layer1_act_pool = ActPool()
        self.layer1_conv = ConvPool(hidden_dimension, hidden_dimension).get_conv(architecture[0])
        self.layer1_norm = NormPool(hidden_dimension).get_norm(architecture[1])
        self.layer1_act = self.layer1_act_pool.get_act(architecture[2])


    def forward(self, x, edge_index):

        x = self.layer1_conv(x, edge_index)
        x = self.layer1_norm(x)
        x = self.layer1_act(x)

        return x

if __name__=="__main__":
    pass