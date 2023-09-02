import torch
import numpy as np
from search_space.mlp import MLP
from search_space.act_pool import ActPool
from search_space.conv_pool import ConvPool
from search_space.norm_pool import NormPool
from Glgcan_utils_cross_validation import cross_validation_graph
from torch_scatter import scatter_add
from sklearn.metrics import accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score
import warnings
warnings.filterwarnings("ignore")

class OneLayerMicroArchitectureBuild(torch.nn.Module):

    def __init__(self,
                 num_node_features,
                 num_classes,
                 hidden_dimension,
                 architecture):

        super(OneLayerMicroArchitectureBuild, self).__init__()

        self.layer1_act_pool = ActPool()
        # build new gnn model based on gnn architecture
        self.pre_process_mlp = MLP(input_dim=num_node_features,
                                   output_dim=hidden_dimension)

        self.post_process_mlp = MLP(input_dim=hidden_dimension,
                                    output_dim=num_classes)

        self.layer1_conv = ConvPool(hidden_dimension, hidden_dimension).get_conv(architecture[0])
        self.layer1_norm = NormPool(hidden_dimension).get_norm(architecture[1])
        self.layer1_act = self.layer1_act_pool.get_act(architecture[2])

    def forward(self, x, edge_index, batch):

        x = self.pre_process_mlp(x)
        x = self.layer1_conv(x, edge_index)
        x = self.layer1_norm(x)
        x = self.layer1_act(x)
        readout_x = self.sum_pooling(x,batch)
        x = self.post_process_mlp(readout_x)

        return x

    def sum_pooling(self,
                    batch_node_embedding_matrix,
                    index):
        graph_embedding = scatter_add(batch_node_embedding_matrix, index, dim=0)

        return graph_embedding

class TwoLayerMicroArchitectureBuild(torch.nn.Module):

    def __init__(self,
                 num_node_features,
                 num_classes,
                 hidden_dimension,
                 architecture):

        super(TwoLayerMicroArchitectureBuild, self).__init__()

        self.layer1_act_pool = ActPool()
        self.layer2_act_pool = ActPool()

        # build new gnn model based on gnn architecture
        self.pre_process_mlp = MLP(input_dim=num_node_features,
                                   output_dim=hidden_dimension)

        self.post_process_mlp = MLP(input_dim=hidden_dimension,
                                    output_dim=num_classes)

        self.layer1_conv = ConvPool(hidden_dimension, hidden_dimension).get_conv(architecture[0])
        self.layer1_norm = NormPool(hidden_dimension).get_norm(architecture[1])
        self.layer1_act = self.layer1_act_pool.get_act(architecture[2])

        self.layer2_conv = ConvPool(hidden_dimension, hidden_dimension).get_conv(architecture[3])
        self.layer2_norm = NormPool(hidden_dimension).get_norm(architecture[4])
        self.layer2_act = self.layer2_act_pool.get_act(architecture[5])

    def forward(self, x, edge_index, batch):

        skip_connection_x_list = []
        x = self.pre_process_mlp(x)

        x = self.layer1_conv(x, edge_index)
        x = self.layer1_norm(x)
        x = self.layer1_act(x)
        skip_connection_x_list.append(x)

        x = self.layer2_conv(x, edge_index)
        x = self.layer2_norm(x)
        x = self.layer2_act(x)
        skip_connection_x_list.append(x)

        skip_connection_x = sum(skip_connection_x_list)
        readout_x = self.sum_pooling(skip_connection_x,batch)
        x = self.post_process_mlp(readout_x)

        return x

    def sum_pooling(self,
                    batch_node_embedding_matrix,
                    index):
        graph_embedding = scatter_add(batch_node_embedding_matrix, index, dim=0)

        return graph_embedding

class MicroArchitectureEvaluation(object):

    def __init__(self, gnn_model_config, graph, device):

        self.num_node_features = gnn_model_config["num_node_features"]
        self.num_classes = gnn_model_config["num_classes"]
        self.hidden_dimension = gnn_model_config["hidden_dimension"]
        self.learning_rate = gnn_model_config["learning_rate"]
        self.weight_decay = gnn_model_config["weight_decay"]
        self.train_epoch = gnn_model_config["train_epoch"]
        self.graph = graph
        self.device = device

    def get_best_validation_score(self, architecture, manner="val", mirco_layer=1):

        torch.cuda.empty_cache()
        if mirco_layer == 1:
            gnn_model = OneLayerMicroArchitectureBuild(num_node_features=self.num_node_features,
                                                       num_classes=self.num_classes,
                                                       hidden_dimension=self.hidden_dimension,
                                                       architecture=architecture).to(self.device)
        elif mirco_layer == 2:
            gnn_model = TwoLayerMicroArchitectureBuild(num_node_features=self.num_node_features,
                                                       num_classes=self.num_classes,
                                                       hidden_dimension=self.hidden_dimension,
                                                       architecture=architecture).to(self.device)

        else:
            raise Exception("Sorry current version don't "
                            "Support this default micro_layer", mirco_layer)

        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        loss_f = torch.nn.CrossEntropyLoss()

        # print('E '
        #       '\t\t Loss '
        #       '\t\t Acc_v '
        #       '\t\t Pre_v '
        #       '\t\t Rec_v '
        #       '\t\t F1_v'
        #       '\t\t Acc_t '
        #       '\t\t Pre_t '
        #       '\t\t Rec_t '
        #       '\t\t F1_t')

        acc_val, p_val, r_val, f1_val, loss = 0, 0, 0, 0, 0
        best_acc_test, best_p_test, best_r_test, best_f1_test, test_loss = 0, 0, 0, 0, 0
        for epoch in range(self.train_epoch):
            gnn_model.train()
            train_y_pre = gnn_model(self.graph.train_x, self.graph.train_edge_index, self.graph.train_batch)
            loss = loss_f(train_y_pre, self.graph.train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            gnn_model.eval()
            val_y_pre = gnn_model(self.graph.val_x, self.graph.val_edge_index, self.graph.val_batch)
            test_y_pre = gnn_model(self.graph.test_x, self.graph.test_edge_index, self.graph.test_batch)

            acc_val, p_val, r_val, f1_val = self.evaluation(val_y_pre, self.graph.val_y)
            acc_test, p_test, r_test, f1_test = self.evaluation(test_y_pre, self.graph.test_y)

            # print('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (
            #     epoch + 1,
            #     loss.item(),
            #     acc_val,
            #     p_val,
            #     r_val,
            #     f1_val,
            #     acc_test,
            #     p_test,
            #     r_test,
            #     f1_test))

            if manner == "test":
                if f1_test > best_f1_test:
                    best_acc_test = acc_test
                    best_p_test = p_test
                    best_r_test = r_test
                    best_f1_test = f1_test


        if manner == "val":
            return acc_val, p_val, r_val, f1_val, loss.item()
        else:
            return best_acc_test, best_p_test, best_r_test, best_f1_test



    def evaluation(self, y_pre, y_true):

        y_pre = y_pre.to("cpu").detach().numpy()
        y_pre = np.argmax(y_pre, axis=1)
        y_true = y_true.to("cpu").detach().numpy()

        acc = accuracy_score(y_true, y_pre)
        precision = precision_score(y_true, y_pre)
        recall = recall_score(y_true, y_pre)
        f1score = f1_score(y_true, y_pre)

        return acc, precision, recall, f1score

    def rank_based_estimation_score(self, gnn_list, val_score_list, top_k):

        gnn_dict = {}

        for key, value in zip(gnn_list, val_score_list):
            gnn_dict[str(key)] = value
        rank_gnn_dict = sorted(gnn_dict.items(), key=lambda x: x[1], reverse=True)

        rank_gnn = []
        rank_gnn_val_score = []

        i = 0
        for key, value in rank_gnn_dict:

            if i == top_k:
                break
            else:
                rank_gnn.append(eval(key))
                rank_gnn_val_score.append(value)
                i += 1
        return rank_gnn, rank_gnn_val_score

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = cross_validation_graph()[0]

    supernet_config = {"input_dimension": graph.node_dim,
                       "hidden_dimension": 128,
                       "output_dimension": graph.label_num,
                       "node_element_dropout_probability": 0.0,
                       "edge_dropout_probability": 0.0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    two_layer_architecture_list = [['SGConv', 'LayerNorm', 'LeakyRelu', 'GATConv', 'InstanceNorm', 'Tanh']]

    one_layer_architecture_list = [['SGConv', 'LayerNorm', 'LeakyRelu']]

    hp_list = [[1e-3, 1e-4, 128]]

    for gnn_architecture, hp in zip(two_layer_architecture_list, hp_list):

        graph_model_config = {"num_node_features": graph.node_dim,
                              "num_classes": graph.label_num,
                              "learning_rate": hp[0],
                              "weight_decay": hp[1],
                              "hidden_dimension": hp[2],
                              "train_epoch": 100}

        model_validation = MicroArchitectureEvaluation(graph_model_config, graph, device)
        model_validation.get_best_validation_score(gnn_architecture, mirco_layer=2)