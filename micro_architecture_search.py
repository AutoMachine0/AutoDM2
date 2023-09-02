import torch
from micro_coupled_supernet import MicroCoupledSuperNet
from Glgcan_utils_cross_validation import cross_validation_graph
from micro_architecture_evaluation import MicroArchitectureEvaluation

class MicroArchitectureSearch(object):

    def __init__(self,
                 operation_candidates_list,
                 supernet_config,
                 archi_param_optim_config,
                 operation_weight_optim_config,
                 device):

        self.supernet = MicroCoupledSuperNet(supernet_config=supernet_config,
                                             archi_param_optim_config=archi_param_optim_config,
                                             operation_weight_optim_config=operation_weight_optim_config,
                                             device=device)

        self.operation_candidates_list = operation_candidates_list
        self.supernet.coupled_supernet_construction_with_operation_candidates(self.operation_candidates_list)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.history_gnn_architecture_list = []

    def search(self, graph, search_epoch, return_top_k):

        print("Micro Architecture Starting")
        print(64*"=")

        for epoch in range(search_epoch):

            y_pre = self.supernet(graph.train_x, graph.train_edge_index, graph.train_batch)

            train_loss = self.loss_function(y_pre, graph.train_y)

            self.supernet.operation_weight_optimizer.zero_grad()
            self.supernet.architecture_parameter_optimizer.zero_grad()

            train_loss.backward()
            self.supernet.operation_weight_optimizer.step()

            y_pre = self.supernet(graph.val_x, graph.val_edge_index, graph.val_batch)

            val_loss = self.loss_function(y_pre,
                                          graph.val_y)

            self.supernet.architecture_parameter_optimizer.zero_grad()
            val_loss.backward()
            self.supernet.architecture_parameter_optimizer.step()

            print(32*"+")
            print("Search Epoch", epoch + 1,
                  "Operation Weight Loss", train_loss.item(),
                  "Architecture Parameter Loss:", val_loss.item())

            best_gnn_architecture = self.best_alpha_gnn_architecture_output(self.supernet.alpha_parameters_list)

            print("Best Micro Architecture:", best_gnn_architecture)

        print(64 * "=")

        if int(return_top_k) <= len(self.history_gnn_architecture_list):
            best_gnn_architecture_candidates = self.history_gnn_architecture_list[-int(return_top_k):]
        else:
            best_gnn_architecture_candidates = self.history_gnn_architecture_list

        print("Sampled Top", return_top_k, "Micro Architecture:")

        for gnn_architecture in best_gnn_architecture_candidates:
            print(gnn_architecture)

        print("Micro Architecture Search Completion")

        return best_gnn_architecture_candidates

    def best_alpha_gnn_architecture_output(self, alpha_parameters_list):

        best_gnn_architecture = []

        for alpha_list, candidate_list in zip(alpha_parameters_list, self.operation_candidates_list):
            alpha_list = alpha_list.cpu().detach().numpy().tolist()
            best_index = alpha_list.index(max(alpha_list))
            best_gnn_architecture.append(candidate_list[best_index])

        if best_gnn_architecture not in self.history_gnn_architecture_list:
            self.history_gnn_architecture_list.append(best_gnn_architecture)

        return best_gnn_architecture

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = cross_validation_graph()[0]

    supernet_config = {"input_dimension": graph.node_dim,
                       "hidden_dimension": 128,
                       "output_dimension": graph.label_num}

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
        best_val_score = gnn_estimation.get_best_validation_score(gnn)
        best_val_score_list.append(best_val_score)

    for score, gnn_architecture in zip(best_val_score_list, promising_gnn_list):
        print(str(gnn_architecture)+" "+str(score))