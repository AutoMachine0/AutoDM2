import torch
import numpy as np
from search_space.mlp import MLP
from torch_scatter import scatter_add
from search_space.act_pool import ActPool
from search_space.conv_pool import ConvPool
from search_space.norm_pool import NormPool
from Glgcan_utils_cross_validation import cross_validation_graph
from sklearn.metrics import accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score
import warnings
warnings.filterwarnings("ignore")

class MicroMacroArchitectureModel(torch.nn.Module):

    def __init__(self,
                 architecture_config,
                 micro_architecture,
                 macro_architecture,
                 device):

        super(MicroMacroArchitectureModel, self).__init__()
        input_dimension = architecture_config["input_dimension"]
        hidden_dimension = architecture_config["hidden_dimension"]
        output_dimension = architecture_config["output_dimension"]
        learning_rate = architecture_config["learning_rate"]
        weight_decay = architecture_config["weight_decay"]

        self.num_state_node = len(macro_architecture) + 1
        self.macro_architecture = macro_architecture
        self.device = device

        self.pre_mlp = MLP(input_dim=input_dimension,
                           output_dim=hidden_dimension).to(self.device)

        self.post_mlp = MLP(input_dim=hidden_dimension,
                            output_dim=output_dimension).to(self.device)

        self.architecture_list = []
        self.architecture_param_list = [{"params": self.pre_mlp.parameters()},
                                        {"params": self.post_mlp.parameters()}]

        for _ in range(len(self.macro_architecture)+1):
            conv = ConvPool(hidden_dimension, hidden_dimension).get_conv(micro_architecture[0]).to(self.device)
            norm = NormPool(hidden_dimension).get_norm(micro_architecture[1]).to(self.device)
            act = ActPool().get_act(micro_architecture[2])
            micro_architecture_cell = [conv, norm, act]
            micro_architecture_cell_param = [{"params": conv.parameters()}, {"params": norm.parameters()}]
            self.architecture_list.append(micro_architecture_cell)
            self.architecture_param_list = self.architecture_param_list + micro_architecture_cell_param

        self.optimizer = torch.optim.Adam(self.architecture_param_list,
                                          lr=learning_rate,
                                          weight_decay=weight_decay)

        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, batch):

        supernet_node_output_manage_list = [[] for _ in range(self.num_state_node - 2)]

        # 处理前n-2个状态节点的输入与输出并保存每个前n-2个状态节点的每一条边的输出
        for node_ith, edge_connection_state_list, micro_cell, state_node_output_manage_list in zip(range(self.num_state_node - 2),
                                                                                                   self.macro_architecture,
                                                                                                   self.architecture_list,
                                                                                                   supernet_node_output_manage_list):

            if node_ith == 0:
                state_node_input = self.pre_mlp(x)
            else:
                state_node_input = []
                for index in range(node_ith):
                    if supernet_node_output_manage_list[index][node_ith - index - 1] != None:
                        state_node_input.append(supernet_node_output_manage_list[index][node_ith - index - 1])
                state_node_input = sum(state_node_input)

            for connection_state_flag in edge_connection_state_list:

                if connection_state_flag == 1:
                    x = micro_cell[0](state_node_input, edge_index)
                    x = micro_cell[1](x)
                    x = micro_cell[2](x)
                else:
                    x = None
                state_node_output_manage_list.append(x)

        # 获取倒数第二个状态节点的输入
        second_to_last_state_node_input = []
        for state_node_output_manage_list in supernet_node_output_manage_list:
            if state_node_output_manage_list[-2] != None:
                second_to_last_state_node_input.append(state_node_output_manage_list[-2])

        # 获取倒数第二个状态节点的输出
        x = self.architecture_list[-2][0](sum(second_to_last_state_node_input), edge_index)
        x = self.architecture_list[-2][1](x)
        second_to_last_state_node_output = self.architecture_list[-2][2](x)

        # 获取最后一个状态节点的输入
        last_state_node_input = []
        for state_node_output_manage_list in supernet_node_output_manage_list:
            if state_node_output_manage_list[-1] != None:
                last_state_node_input.append(state_node_output_manage_list[-1])
        last_state_node_input.append(second_to_last_state_node_output)

        # 获取最后一个状态节点的输出
        x = self.architecture_list[-1][0](sum(last_state_node_input), edge_index)
        x = self.architecture_list[-1][1](x)
        last_state_node_output = self.architecture_list[-1][2](x)

        graph_embedding = self.sum_pooling(last_state_node_output, batch)
        y = self.post_mlp(graph_embedding)

        return y

    def sum_pooling(self,
                    batch_node_embedding_matrix,
                    index):

        graph_embedding = scatter_add(batch_node_embedding_matrix, index, dim=0)

        return graph_embedding

class MicroMacroArchitectureModelTrainTest(object):

    def __init__(self,
                 architecture_config,
                 micro_architecture,
                 macro_architecture,
                 device):

        self.architecture_config = architecture_config
        self.micro_architecture = micro_architecture
        self.macro_architecture = macro_architecture
        self.device = device
        self.micro_macro_architecture = MicroMacroArchitectureModel(architecture_config=self.architecture_config,
                                                                    micro_architecture=self.micro_architecture,
                                                                    macro_architecture=self.macro_architecture,
                                                                    device=self.device)

    def train_test(self, graph, training_epoch, manner="best_acc"):

        best_acc_test, best_p_test, best_r_test, best_f1_test = 0, 0, 0, 0
        best_acc_val, best_p_val, best_r_val, best_f1_val = 0, 0, 0, 0
        best_epoch = 0

        for epoch in range(training_epoch):
            self.micro_macro_architecture.train()
            y_pre = self.micro_macro_architecture(graph.train_x, graph.train_edge_index, graph.train_batch)
            train_loss = self.micro_macro_architecture.loss_function(y_pre, graph.train_y)

            self.micro_macro_architecture.optimizer.zero_grad()
            train_loss.backward()
            self.micro_macro_architecture.optimizer.step()

            self.micro_macro_architecture.eval()

            val_y_pre = self.micro_macro_architecture(graph.val_x, graph.val_edge_index, graph.val_batch)
            test_y_pre = self.micro_macro_architecture(graph.test_x, graph.test_edge_index, graph.test_batch)

            acc_val, p_val, r_val, f1_val = self.evaluation(val_y_pre, graph.val_y)
            acc_test, p_test, r_test, f1_test = self.evaluation(test_y_pre, graph.test_y)

            print('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (
                epoch + 1,
                train_loss.item(),
                acc_val,
                p_val,
                r_val,
                f1_val,
                acc_test,
                p_test,
                r_test,
                f1_test))

            if manner == "test_acc":
                if best_acc_test < acc_test:
                    best_acc_test = acc_test
                    best_p_test = p_test
                    best_r_test = r_test
                    best_f1_test = f1_test
                    best_epoch = epoch + 1

            elif manner == "test_f1":
                if best_f1_test < f1_test:
                    best_acc_test = acc_test
                    best_p_test = p_test
                    best_r_test = r_test
                    best_f1_test = f1_test
                    best_epoch = epoch + 1

            elif manner == "val_f1+val_acc":
                if best_f1_val+best_acc_val < f1_val+acc_val:
                    best_f1_val = f1_val
                    best_acc_val = acc_val

                    best_acc_test = acc_test
                    best_p_test = p_test
                    best_r_test = r_test
                    best_f1_test = f1_test
                    best_epoch = epoch + 1

            elif manner == "val_f1":
                if best_f1_val <= f1_val:
                    best_f1_val = f1_val

                    best_acc_test = acc_test
                    best_p_test = p_test
                    best_r_test = r_test
                    best_f1_test = f1_test
                    best_epoch = epoch + 1

            elif manner == "val_acc":
                if best_acc_val <= acc_val:
                    best_acc_val = acc_val

                    best_acc_test = acc_test
                    best_p_test = p_test
                    best_r_test = r_test
                    best_f1_test = f1_test
                    best_epoch = epoch + 1

            elif manner == "val_recall":
                if best_r_val <= r_val:
                    best_r_val = r_val

                    best_acc_test = acc_test
                    best_p_test = p_test
                    best_r_test = r_test
                    best_f1_test = f1_test
                    best_epoch = epoch + 1

            elif manner == "last_epoch":
                best_acc_test = acc_test
                best_p_test = p_test
                best_r_test = r_test
                best_f1_test = f1_test
                best_epoch = epoch + 1


        if manner == "validation":

            return acc_val, p_val, r_val, f1_val

        return best_acc_test, best_p_test, best_r_test, best_f1_test, best_epoch

    def evaluation(self, y_pre, y_true):
        y_pre = y_pre.to("cpu").detach().numpy()
        y_pre = np.argmax(y_pre, axis=1)
        y_true = y_true.to("cpu").detach().numpy()

        acc = accuracy_score(y_true, y_pre)
        precision = precision_score(y_true, y_pre)
        recall = recall_score(y_true, y_pre)
        f1score = f1_score(y_true, y_pre)

        return acc, precision, recall, f1score

if __name__=="__main__":

    macro_architecture = [[1, 1, 0, 1], [1, 0, 1], [1, 1], [1]]
    micro_architecture = ["GCNConv", "GraphNorm", "Relu"]

    graph = cross_validation_graph()[0]

    architecture_config = {"input_dimension": graph.node_dim,
                           "hidden_dimension": 128,
                           "output_dimension": graph.label_num,
                           "learning_rate": 0.01,
                           "weight_decay": 0.001}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_epoch = 50

    model_test = MicroMacroArchitectureModelTrainTest(architecture_config=architecture_config,
                                                      macro_architecture=macro_architecture,
                                                      micro_architecture=micro_architecture,
                                                      device=device)

    model_test.train_test(graph, training_epoch)