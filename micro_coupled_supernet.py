import torch
import torch.nn as nn
import torch.nn.functional as F
from search_space.mlp import MLP
from torch.autograd import Variable
from Glgcan_utils_cross_validation import cross_validation_graph
from torch_scatter import scatter_add, scatter_max, scatter_mean
from search_space_with_forward.act_pool import ActPool
from search_space_with_forward.conv_pool import ConvPool
from search_space_with_forward.norm_pool import NormPool

class MicroCoupledSuperNet(torch.nn.Module):

    def __init__(self,
                 supernet_config,
                 archi_param_optim_config,
                 operation_weight_optim_config,
                 device):

        super(MicroCoupledSuperNet, self).__init__()

        self.input_dimension = supernet_config["input_dimension"]
        self.hidden_dimension = supernet_config["hidden_dimension"]
        self.output_dimension = supernet_config["output_dimension"]

        self.mic_archi_param_learning_rate = archi_param_optim_config["mic_archi_param_learning_rate"]
        self.mic_archi_param_weight_decay = archi_param_optim_config["mic_archi_param_weight_decay"]
        self.mic_archi_param_temperature = archi_param_optim_config["mic_archi_param_temperature"]

        self.mic_operation_weight_learning_rate = operation_weight_optim_config["mic_operation_weight_learning_rate"]
        self.mic_operation_weight_weight_decay = operation_weight_optim_config["mic_operation_weight_weight_decay"]

        self.device = device
        # coupled supernet candidate
        self.conv_candidate = ConvPool().candidate_list
        self.norm_candidate = NormPool().candidate_list
        self.act_candidate = ActPool().candidate_list

    def coupled_supernet_construction_with_operation_candidates(self, operation_candidates_list):

        # pre process mlp initialization
        self.pre_process_mlp = MLP(input_dim=self.input_dimension,
                                   output_dim=self.hidden_dimension).to(self.device)
        # post process mlp initialization
        self.post_process_mlp = MLP(input_dim=self.hidden_dimension,
                                    output_dim=self.output_dimension).to(self.device)

        self.architecture_parameter_construction_with_operation_candidate(operation_candidates_list)

        self.inference_flow_with_mix_operation = []

        operation_weights = []

        for operation_candidate in operation_candidates_list:

            mix_operation = []

            for operation in operation_candidate:

                if operation in self.conv_candidate:
                    operation_obj = ConvPool(input_dim=self.hidden_dimension,
                                             output_dim=self.hidden_dimension,
                                             conv_name=operation).to(self.device)
                elif operation in self.norm_candidate:
                    operation_obj = NormPool(input_dim=self.hidden_dimension,
                                             norm_name=operation).to(self.device)
                elif operation in self.act_candidate:
                    operation_obj = ActPool(act_name=operation).to(self.device)
                else:
                    raise Exception("Sorry current version don't "
                                    "Support this operation", operation)

                operation_weights.append({"params": operation_obj.parameters()})

                mix_operation.append(operation_obj)

            self.inference_flow_with_mix_operation.append(mix_operation)

        self.operation_weight_optimizer = torch.optim.Adam(operation_weights,
                                                           lr=self.mic_operation_weight_learning_rate,
                                                           weight_decay=self.mic_operation_weight_weight_decay)

    def architecture_parameter_construction_with_operation_candidate(self, operation_candidates_list):

        self.alpha_parameters_list = []

        for operation_candidate in operation_candidates_list:

            num_operation_candidates = len(operation_candidate)
            alpha_parameters = Variable(nn.init.uniform_(torch.Tensor(num_operation_candidates))).to(self.device)
            alpha_parameters.requires_grad = True
            nn.init.uniform_(alpha_parameters)
            self.alpha_parameters_list.append(alpha_parameters)

        self.architecture_parameter_optimizer = torch.optim.Adam(self.alpha_parameters_list,
                                                                 lr=self.mic_archi_param_learning_rate,
                                                                 weight_decay=self.mic_archi_param_weight_decay)

    def forward(self, x, edge_index, batch):

        x = self.pre_process_mlp(x)
        gnn_layer_output_list = []

        for mix_operation, alpha_operation in zip(self.inference_flow_with_mix_operation, self.alpha_parameters_list):

            alpha_operation = F.softmax(alpha_operation/self.mic_archi_param_temperature, dim=-1)
            operation_output_list = []
            operation = None

            for operation, alpha in zip(mix_operation, alpha_operation):
                if type(operation).__name__ == "ConvPool":
                    operation_output_list.append(operation(x, edge_index) * alpha)
                    continue
                elif type(operation).__name__ == "NormPool":
                    operation_output_list.append(operation(x) * alpha)
                    continue
                elif type(operation).__name__ == "ActPool":
                    operation_output_list.append(operation(x) * alpha)
                    continue
                else:
                    raise Exception("Sorry current version don't "
                                    "Support this default operation", type(operation).__name__)
            # calculate mixed operation output for next mixed operation input
            if type(operation).__name__ != "ActPool":
                x = sum(operation_output_list)
            else:
                x = sum(operation_output_list)
                gnn_layer_output_list.append(x)

        skip_connection_x = sum(gnn_layer_output_list)
        graph_read_out = self.sum_pooling(skip_connection_x, batch)
        graph_embedding = self.post_process_mlp(graph_read_out)

        return graph_embedding

    def sum_pooling(self,
                    batch_node_embedding_matrix,
                    index):

        graph_embedding = scatter_add(batch_node_embedding_matrix, index, dim=0)

        return graph_embedding

    def mean_pooling(self,
                     batch_node_embedding_matrix,
                     index):

        graph_embedding = scatter_mean(batch_node_embedding_matrix, index, dim=0)

        return graph_embedding

    def max_pooling(self,
                    batch_node_embedding_matrix,
                    index):

        graph_embedding, _ = scatter_max(batch_node_embedding_matrix, index, dim=0)

        return graph_embedding

if __name__=="__main__":

    data_name = "Computers"
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
                                    "SGConv", "GINConv"],
                                   ["GraphNorm", "InstanceNorm",
                                    "LayerNorm", "BatchNorm"],
                                   ["Elu", "LeakyRelu",
                                    "Relu", "Sigmoid",
                                    "Softplus", "Tanh"]]

    archi_param_optim_config = {"mic_archi_param_learning_rate": 0.1,
                                "mic_archi_param_weight_decay": 0.01}

    operation_weight_optim_config = {"mic_operation_weight_learning_rate": 0.001,
                                     "mic_operation_weight_weight_decay": 0.0001}

    my_supernet = MicroCoupledSuperNet(supernet_config=supernet_config,
                                       archi_param_optim_config=archi_param_optim_config,
                                       operation_weight_optim_config=operation_weight_optim_config,
                                       device=device)

    my_supernet.coupled_supernet_construction_with_operation_candidates(operation_candidates_list_1)
    loss_f = torch.nn.CrossEntropyLoss()

    search_epoch = 100
    for epoch in range(search_epoch):
        y_pre = my_supernet(graph.train_x, graph.train_edge_index, graph.train_batch)

        train_loss = loss_f(y_pre, graph.train_y)

        my_supernet.operation_weight_optimizer.zero_grad()
        my_supernet.architecture_parameter_optimizer.zero_grad()

        train_loss.backward()
        my_supernet.operation_weight_optimizer.step()
        y_pre = my_supernet(graph.val_x, graph.val_edge_index, graph.val_batch)
        val_loss = loss_f(y_pre, graph.val_y)

        my_supernet.architecture_parameter_optimizer.zero_grad()
        val_loss.backward()
        my_supernet.architecture_parameter_optimizer.step()

        print(32 * "+")
        print("Search Epoch", epoch + 1,
              "Operation Weight Loss", train_loss.item(),
              "Architecture Parameter Loss:", val_loss.item())