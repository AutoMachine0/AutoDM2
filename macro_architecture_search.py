import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from search_space.mlp import MLP
from torch.autograd import Variable
from torch_scatter import scatter_add
from search_space.act_pool import ActPool
from search_space.conv_pool import ConvPool
from search_space.norm_pool import NormPool
from Glgcan_utils_cross_validation import cross_validation_graph
from micro_architecture_search import MicroArchitectureSearch
from micro_architecture_evaluation import MicroArchitectureEvaluation
from identity_operation import IdentityOperation_1, IdentityOperation_2
from sklearn.metrics import accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score


class MarcoArchitectureWithDepthModel(torch.nn.Module):

    def __init__(self, macro_architecture_depth, macro_architecture_config, device):

        super(MarcoArchitectureWithDepthModel, self).__init__()

        self.learning_rate = macro_architecture_config["learning_rate"]
        self.weight_decay = macro_architecture_config["weight_decay"]
        self.input_dimension = macro_architecture_config["input_dimension"]
        self.hidden_dimension = macro_architecture_config["hidden_dimension"]
        self.output_dimension = macro_architecture_config["output_dimension"]
        self.micro_architecture = macro_architecture_config["micro_architecture"]
        self.device = device

        self.pre_mlp = MLP(input_dim=self.input_dimension,
                           output_dim=self.hidden_dimension).to(self.device)
        self.post_mlp = MLP(input_dim=self.hidden_dimension,
                            output_dim=self.output_dimension).to(self.device)

        self.architecture_list = []
        self.architecture_param_list = [{"params": self.pre_mlp.parameters()},
                                        {"params": self.post_mlp.parameters()}]

        for _ in range(macro_architecture_depth):
            conv = ConvPool(self.hidden_dimension, self.hidden_dimension).get_conv(self.micro_architecture[0]).to(
                self.device)
            norm = NormPool(self.hidden_dimension).get_norm(self.micro_architecture[1]).to(self.device)
            act = ActPool().get_act(self.micro_architecture[2])
            micro_architecture_cell = [conv, norm, act]
            micro_architecture_cell_param = [{"params": conv.parameters()}, {"params": norm.parameters()}]
            self.architecture_list.append(micro_architecture_cell)
            self.architecture_param_list = self.architecture_param_list + micro_architecture_cell_param

        self.optimizer = torch.optim.Adam(self.architecture_param_list,
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)

    def forward(self, x, edge_index, batch):

        x = self.pre_mlp(x)
        for micro_architecture_cell in self.architecture_list:
            x = micro_architecture_cell[0](x, edge_index)
            x = micro_architecture_cell[1](x)
            x = micro_architecture_cell[2](x)
        x = self.post_mlp(x)
        y_pre = self.sum_pooling(x, batch)
        return y_pre

    def sum_pooling(self,
                    batch_node_embedding_matrix,
                    index):

        graph_embedding = scatter_add(batch_node_embedding_matrix, index, dim=0)

        return graph_embedding


class MarcoArchitectureDepthSearch(torch.nn.Module):

    def __init__(self, depth, macro_architecture_config, device, graph):
        super(MarcoArchitectureDepthSearch, self).__init__()
        self.graph = graph
        self.depth = depth
        self.device = device
        self.macro_architecture_config = macro_architecture_config
        self.training_epoch = macro_architecture_config["training_epoch"]
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.macro_architecture_model = MarcoArchitectureWithDepthModel(macro_architecture_depth=self.depth,
                                                                        macro_architecture_config=macro_architecture_config,
                                                                        device=self.device)

    def macro_architecture_depth_search(self, feedback="acc"):

        val_score_list = []

        for depth in range(self.depth):

            print("Evaluating Macro Architecture with Depth:", depth+1)
            macro_architecture_model = MarcoArchitectureWithDepthModel(macro_architecture_depth=depth+1,
                                                                       macro_architecture_config=self.macro_architecture_config,
                                                                       device=self.device)

            acc, pre, recall, f1, train_loss = self.macro_architecture_evaluation(macro_architecture_model)

            if feedback == "acc":
                best_val = acc
            elif feedback == "pre":
                best_val = pre
            elif feedback == "recall":
                best_val = recall
            elif feedback == "f1":
                best_val = f1
            elif feedback == "acc+f1":
                best_val = acc+f1
            else:
                best_val = train_loss

            val_score_list.append(best_val)

        if feedback != "train_loss":
            max_value = max(val_score_list)
            tup = [(i, val_score_list[i]) for i in range(len(val_score_list))]
            best_depth_list = [i for i, n in tup if n == max_value]
        else:
            min_value = min(val_score_list)
            tup = [(i, val_score_list[i]) for i in range(len(val_score_list))]
            best_depth_list = [i for i, n in tup if n == min_value]

        best_depth = best_depth_list[-1] + 1

        print("Best Val Score with Different Depth:", val_score_list)
        print("Best Depth:", best_depth)

        return best_depth

    def macro_architecture_evaluation(self, macro_architecture_model):
        acc_val, p_val, r_val, f1_val, train_loss = 0, 0, 0, 0, 0
        for epoch in range(self.training_epoch):

            macro_architecture_model.train()

            y_pre = macro_architecture_model(self.graph.train_x,
                                             self.graph.train_edge_index,
                                             self.graph.train_batch)
            train_loss = self.loss_function(y_pre, self.graph.train_y)
            macro_architecture_model.optimizer.zero_grad()
            train_loss.backward()
            macro_architecture_model.optimizer.step()

            macro_architecture_model.eval()
            val_y_pre = macro_architecture_model(self.graph.val_x, self.graph.val_edge_index, self.graph.val_batch)
            test_y_pre = macro_architecture_model(self.graph.test_x, self.graph.test_edge_index, self.graph.test_batch)

            acc_val, p_val, r_val, f1_val = self.evaluation(val_y_pre, self.graph.val_y)
            acc_test, p_test, r_test, f1_test = self.evaluation(test_y_pre, self.graph.test_y)

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

        return acc_val, p_val, r_val, f1_val, train_loss.item()

    def evaluation(self, y_pre, y_true):
        y_pre = y_pre.to("cpu").detach().numpy()
        y_pre = np.argmax(y_pre, axis=1)
        y_true = y_true.to("cpu").detach().numpy()

        acc = accuracy_score(y_true, y_pre)
        precision = precision_score(y_true, y_pre)
        recall = recall_score(y_true, y_pre)
        f1score = f1_score(y_true, y_pre)

        return acc, precision, recall, f1score


class MacroCoupledSupernet(torch.nn.Module):

    def __init__(self,
                 num_state_node,
                 archi_param_optim_config,
                 identity_op_config,
                 identity_param_optim_config,
                 pre_post_mlp_config,
                 device):

        super(MacroCoupledSupernet, self).__init__()
        self.device = device

        self.num_state_node = num_state_node
        self.archi_param_learning_rate = archi_param_optim_config["archi_param_learning_rate"]
        self.archi_param_weight_decay = archi_param_optim_config["archi_param_weight_decay"]
        self.archi_param_temperature = archi_param_optim_config["archi_param_temperature"]

        self.architecture_parameter_construction_with_num_state_node()

        self.identity_op_dim = identity_op_config["identity_op_dim"]
        self.identity_op_micro_architecture = identity_op_config["identity_op_micro_architecture"]

        self.identity_param_learning_rate = identity_param_optim_config["identity_param_learning_rate"]
        self.identity_param_weight_decay = identity_param_optim_config["identity_param_weight_decay"]

        self.identity_op_construction_with_micro_architecture()

        self.pre_mlp_input_dim = pre_post_mlp_config["pre_mlp_input_dim"]
        self.pre_mlp_output_dim = pre_post_mlp_config["pre_mlp_output_dim"]
        self.post_mlp_input_dim = pre_post_mlp_config["post_mlp_input_dim"]
        self.post_mlp_output_dim = pre_post_mlp_config["post_mlp_output_dim"]

        self.pre_mlp = MLP(input_dim=self.pre_mlp_input_dim,
                           output_dim=self.pre_mlp_output_dim).to(self.device)

        self.post_mlp = MLP(input_dim=self.post_mlp_input_dim,
                            output_dim=self.post_mlp_output_dim).to(self.device)

    def identity_op_construction_with_micro_architecture(self):

        self.identity_op = IdentityOperation_1(hidden_dimension=self.identity_op_dim,
                                               architecture=self.identity_op_micro_architecture).to(self.device)

        self.identity_op_optimizer = torch.optim.Adam(self.identity_op.parameters(),
                                                      lr=self.identity_param_learning_rate,
                                                      weight_decay=self.identity_param_weight_decay)
        self.identity_op_loss_function = torch.nn.CrossEntropyLoss()

    def architecture_parameter_construction_with_num_state_node(self):

        if self.num_state_node == 1 or self.num_state_node == 2:
            return None

        self.supernet_alpha_parameters_list = []

        for state_node in range(self.num_state_node - 1):

            if state_node == self.num_state_node - 2:
                break
            state_node_alpha_parameters_list = []

            for direction_edge_num in range(self.num_state_node - (state_node + 2)):
                # each direction edge has two alpha parameters
                alpha_parameters = Variable(nn.init.uniform_(torch.Tensor(2)))
                alpha_parameters.requires_grad = True
                nn.init.uniform_(alpha_parameters)
                state_node_alpha_parameters_list.append(alpha_parameters)
            # add all direction edge alpha parameters of this state node into supernet_alpha_parameters_list
            self.supernet_alpha_parameters_list.append({"params": state_node_alpha_parameters_list})

        self.architecture_parameter_optimizer = torch.optim.Adam(self.supernet_alpha_parameters_list,
                                                                 lr=self.archi_param_learning_rate,
                                                                 weight_decay=self.archi_param_weight_decay)
        self.architecture_parameter_loss_function = torch.nn.CrossEntropyLoss()

    def zero_op(self):

        return torch.tensor(0.0)

    def forward(self, x, edge_index, batch):

        supernet_node_output_manage_list = [[] for _ in range(self.num_state_node - 2)]

        # 处理前n-1个状态节点的输入与输出并保存每个前n-1个状态节点的每一条边的输出
        for node_ith, state_node_alpha_parameters_list, state_node_output_manage_list in zip(range(self.num_state_node - 2),
                                                                                             self.supernet_alpha_parameters_list,
                                                                                             supernet_node_output_manage_list):
            if node_ith == 0:
                state_node_input = self.pre_mlp(x)
            else:
                state_node_input = []
                for index in range(node_ith):
                    state_node_input.append(supernet_node_output_manage_list[index][node_ith - index - 1])
                state_node_input = sum(state_node_input)

            for alpha_parameters in state_node_alpha_parameters_list["params"]:

                alpha_parameters = F.softmax(alpha_parameters/self.archi_param_temperature, dim=-1)

                mixed_operation = [self.zero_op() * alpha_parameters[0],
                                   self.identity_op(state_node_input, edge_index) * alpha_parameters[1]]
                state_node_output_manage_list.append(sum(mixed_operation))

            # 给下一个状态节点输入增加一条来自本状态节点的必连的边作为输入
            state_node_output_manage_list.insert(0, self.identity_op(state_node_input, edge_index))
        # 获取倒数第二个状态节点的输入
        second_to_last_state_node_input = []
        for state_node_output_manage_list in supernet_node_output_manage_list:
            second_to_last_state_node_input.append(state_node_output_manage_list[-2])

        # 获取倒数第二个状态节点的输出
        second_to_last_state_node_output = self.identity_op(sum(second_to_last_state_node_input), edge_index)

        # 获取最后一个状态节点的输入
        last_state_node_input = []
        for state_node_output_manage_list in supernet_node_output_manage_list:
            last_state_node_input.append(state_node_output_manage_list[-1])
        last_state_node_input.append(second_to_last_state_node_output)

        # 获取最后一个状态节点的输出
        last_state_node_output = self.identity_op(sum(last_state_node_input), edge_index)

        graph_embedding = self.sum_pooling(last_state_node_output, batch)
        y = self.post_mlp(graph_embedding)

        return y

    def sum_pooling(self,
                    batch_node_embedding_matrix,
                    index):

        graph_embedding = scatter_add(batch_node_embedding_matrix, index, dim=0)

        return graph_embedding

class MacroArchitectureSkipConnectionSearch(object):

    def __init__(self,
                 num_state_node,
                 archi_param_optim_config,
                 identity_op_config,
                 identity_param_optim_config,
                 pre_post_mlp_config,
                 device):

        self.num_state_node = num_state_node

        self.supenet = MacroCoupledSupernet(num_state_node,
                                            archi_param_optim_config,
                                            identity_op_config,
                                            identity_param_optim_config,
                                            pre_post_mlp_config,
                                            device)

        self.history_macro_architecture_list = []

    def search(self, graph, search_epoch, return_top_k):

        if self.num_state_node == 1 or self.num_state_node == 2:
            return []

        print("Marco Architecture Search Starting")
        print(64*"+")

        for epoch in range(search_epoch):

            y_pre = self.supenet(graph.train_x, graph.train_edge_index, graph.train_batch)
            train_loss = self.supenet.identity_op_loss_function(y_pre, graph.train_y)

            self.supenet.identity_op_optimizer.zero_grad()
            self.supenet.architecture_parameter_optimizer.zero_grad()

            train_loss.backward()
            self.supenet.identity_op_optimizer.step()

            y_pre = self.supenet(graph.val_x, graph.val_edge_index, graph.val_batch)
            val_loss = self.supenet.architecture_parameter_loss_function(y_pre, graph.val_y)

            self.supenet.architecture_parameter_optimizer.zero_grad()
            val_loss.backward()

            self.supenet.architecture_parameter_optimizer.step()
            best_macro_architecture = self.best_macro_architecture_output(self.supenet.supernet_alpha_parameters_list)

            for micro_cell in best_macro_architecture:
                micro_cell.insert(0, 1)

            best_macro_architecture.append([1])

            print("Search Epoch", epoch+1,
                  "Micro Architecture Parameter Train Loss", train_loss.item(),
                  "Macro Architecture Parameter Validation Loss", val_loss.item())
            print("Macro Architecture:", best_macro_architecture)
            if best_macro_architecture not in self.history_macro_architecture_list:
                self.history_macro_architecture_list.append(best_macro_architecture)

        return self.history_macro_architecture_list[-return_top_k:]

    def best_macro_architecture_output(self, supernet_alpha_parameters_list):

        best_macro_architecture = []
        for state_node_alpha_parameters in supernet_alpha_parameters_list:
            state_node_list = []
            for one_edge_alpha_parameters in state_node_alpha_parameters["params"]:
                one_edge_alpha_parameters = one_edge_alpha_parameters.cpu().detach().numpy().tolist()
                best_index = one_edge_alpha_parameters.index(max(one_edge_alpha_parameters))
                state_node_list.append(best_index)
            best_macro_architecture.append(state_node_list)

        return best_macro_architecture

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = cross_validation_graph()[0]

    supernet_config = {"input_dimension": graph.node_dim,
                       "hidden_dimension": 128,
                       "output_dimension": graph.label_num,
                       "node_element_dropout_probability": 0.0,
                       "edge_dropout_probability": 0.0}

    operation_candidates_list_2 = [["GCNConv", "SAGEConv",
                                    "GATConv", "GraphConv",
                                    "SGConv", "GINConv"],
                                   ["GraphNorm", "InstanceNorm",
                                    "LayerNorm", "BatchNorm"],
                                   ["Elu", "LeakyRelu",
                                    "Relu", "Sigmoid",
                                    "Softplus", "Tanh"],
                                   ["GCNConv", "SAGEConv",
                                    "GATConv", "GraphConv",
                                    "SGConv", "GINConv"],
                                   ["GraphNorm", "InstanceNorm",
                                    "LayerNorm", "BatchNorm"],
                                   ["Elu", "LeakyRelu",
                                    "Relu", "Sigmoid",
                                    "Softplus", "Tanh"]]

    operation_candidates_list_1 = [["GCNConv", "SAGEConv",
                                    "GATConv", "GraphConv",
                                    "TAGConv", "ARMAConv",
                                    "SGConv", "HyperGraphConv",
                                    "ClusterGCNConv", "GINConv"],
                                   ["GraphNorm", "InstanceNorm",
                                    "LayerNorm", "BatchNorm",
                                    "LinearNorm"],
                                   ["Elu", "LeakyRelu",
                                    "Relu", "Relu6",
                                    "Sigmoid", "Softplus",
                                    "Tanh", "Linear"]]

    archi_param_optim_config = {"archi_param_learn_rate": 0.1,
                                "archi_param_weight_decay": 0.01}

    operation_weight_optim_config = {"operation_weight_learn_rate": 0.001,
                                     "operation_weight_weight_decay": 0.0001}

    mydarts = MicroArchitectureSearch(operation_candidates_list=operation_candidates_list_1,
                                      supernet_config=supernet_config,
                                      operation_weight_optim_config=operation_weight_optim_config,
                                      archi_param_optim_config=archi_param_optim_config,
                                      device=device)

    promising_gnn_list = mydarts.search(graph, 100, 10)

    hp = [0.001, 0.0001, 128]
    train_epoch = 100

    graph_model_config = {"num_node_features": graph.node_dim,
                          "num_classes": graph.label_num,
                          "learn_rate": hp[0],
                          "weight_decay": hp[1],
                          "hidden_dimension": hp[2],
                          "train_epoch": train_epoch}

    gnn_estimation = MicroArchitectureEvaluation(graph_model_config, graph, device)

    best_val_score_list = []
    for gnn in promising_gnn_list:
        best_val_score,_,_,_ = gnn_estimation.get_best_validation_score(gnn)
        best_val_score_list.append(best_val_score)

    best_val_index = best_val_score_list.index(max(best_val_score_list))
    micro_architecture = promising_gnn_list[best_val_index]

    # micro_architecture = ["GCNConv", "GraphNorm", "Relu"]

    depth = 10
    macro_architecture_config = {"learning_rate": 0.01,
                                 "weight_decay": 0.001,
                                 "input_dimension": graph.node_dim,
                                 "hidden_dimension": 128,
                                 "output_dimension": graph.label_num,
                                 "micro_architecture": micro_architecture,
                                 "training_epoch": 50}

    macro_architecture_depth = MarcoArchitectureDepthSearch(depth=depth,
                                                            macro_architecture_config=macro_architecture_config,
                                                            device=device,
                                                            graph=graph)

    best_depth = macro_architecture_depth.macro_architecture_depth_search()

    num_state_node = best_depth

    micro_architecture = ["GCNConv", "GraphNorm", "Relu"]

    pre_post_mlp_config = {"pre_mlp_input_dim": graph.node_dim,
                           "pre_mlp_output_dim": 128,
                           "post_mlp_input_dim": 128,
                           "post_mlp_output_dim": graph.label_num}

    identity_op_config = {"identity_op_dim": 128,
                          "identity_op_node_element_dropout_probability": 0.0,
                          "identity_op_edge_dropout_probability": 0.0,
                          "identity_op_micro_architecture": micro_architecture}

    archi_param_optim_config = {"archi_param_learning_rate": 0.1,
                                "archi_param_weight_decay": 0.001,
                                "archi_param_temperature": 0.1}

    identity_param_optim_config = {"identity_param_learning_rate": 0.01,
                                   "identity_param_weight_decay": 0.001}

    macro_search = MacroArchitectureSkipConnectionSearch(num_state_node,
                                                         archi_param_optim_config,
                                                         identity_op_config,
                                                         identity_param_optim_config,
                                                         pre_post_mlp_config,
                                                         device)

    search_epoch = 100
    macro_architecture = macro_search.search(graph, search_epoch, 10)
    print("Micro Architecture:", micro_architecture)
    print("Macro Architecture:", macro_architecture)