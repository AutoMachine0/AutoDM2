import torch
from micro_architecture_search import MicroArchitectureSearch
from Glgcan_utils_cross_validation import cross_validation_graph
from micro_architecture_evaluation import MicroArchitectureEvaluation
from architecture_build_test import MicroMacroArchitectureModelTrainTest
from macro_architecture_search import MacroArchitectureSkipConnectionSearch, MarcoArchitectureDepthSearch


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = cross_validation_graph()[0]

    feedback_mic_archi_evaluation = "acc+f1"
    mic_archi_param_learning_rate = 0.1
    mic_archi_param_weight_decay = 0.001
    mic_archi_param_temperature = 2.5

    hidden_dimension = 128
    mic_operation_weight_learning_rate = 0.01
    mic_operation_weight_weight_decay = 0.0005

    mic_archi_search_epoch = 100
    mic_promising_archi_num = 5
    mic_archi_select_train_epoch = 100

    feedback_mac_depth_evaluation = "acc+f1"
    mac_archi_depth_explore = 10
    mac_archi_depth_explore_train_epoch = 50

    mac_archi_param_learning_rate = 0.1
    mac_archi_param_weight_decay = 0.001
    mac_archi_param_temperature = 2.0

    mac_archi_skip_connection_search_epoch = 350
    mac_promising_skip_connection_archi_num = 5

    final_architecture_training_epoch = 100

    # micro architecture search based on gradient

    supernet_config = {"input_dimension": graph.node_dim,
                       "hidden_dimension": hidden_dimension,
                       "output_dimension": graph.label_num}

    operation_candidates_list_1 = [["GCNConv", "SAGEConv",
                                    "GraphConv", "ARMAConv"],
                                   ["GraphNorm", "InstanceNorm",
                                    "LayerNorm", "BatchNorm",
                                    "LinearNorm"],
                                   ["Elu", "LeakyRelu",
                                    "Relu", "Relu6",
                                    "Sigmoid", "Softplus",
                                    "Tanh", "Linear"]]

    archi_param_optim_config = {"mic_archi_param_learning_rate": mic_archi_param_learning_rate,
                                "mic_archi_param_weight_decay": mic_archi_param_weight_decay,
                                "mic_archi_param_temperature": mic_archi_param_temperature}

    operation_weight_optim_config = {"mic_operation_weight_learning_rate": mic_operation_weight_learning_rate,
                                     "mic_operation_weight_weight_decay": mic_operation_weight_weight_decay}

    micro_architecture = MicroArchitectureSearch(operation_candidates_list=operation_candidates_list_1,
                                                 supernet_config=supernet_config,
                                                 operation_weight_optim_config=operation_weight_optim_config,
                                                 archi_param_optim_config=archi_param_optim_config,
                                                 device=device)

    promising_gnn_list = micro_architecture.search(graph,
                                                   mic_archi_search_epoch,
                                                   mic_promising_archi_num)

    # # promising micro architecture evaluation
    graph_model_config = {"num_node_features": graph.node_dim,
                          "num_classes": graph.label_num,
                          "learning_rate": mic_operation_weight_learning_rate,
                          "weight_decay": mic_operation_weight_weight_decay,
                          "hidden_dimension": hidden_dimension,
                          "train_epoch": mic_archi_select_train_epoch}

    gnn_estimation = MicroArchitectureEvaluation(graph_model_config, graph, device)

    best_val_score_list = []
    for gnn in promising_gnn_list:
        acc, pre, recall, f1, train_loss = gnn_estimation.get_best_validation_score(gnn)

        if feedback_mic_archi_evaluation == "acc":
            score = acc
        elif feedback_mic_archi_evaluation == "pre":
            score = pre
        elif feedback_mic_archi_evaluation == "recall":
            score = recall
        elif feedback_mic_archi_evaluation == "f1":
            score = f1
        elif feedback_mic_archi_evaluation == "acc+f1":
            score = acc + f1
        else:
            score = train_loss

        best_val_score_list.append(score)

    if feedback_mic_archi_evaluation != "train_loss":
         best_val_index = best_val_score_list.index(max(best_val_score_list))
    else:
         best_val_index = best_val_score_list.index(min(best_val_score_list))

    micro_architecture = promising_gnn_list[best_val_index]
    print(best_val_score_list)
    print(micro_architecture)
    print(promising_gnn_list)

    # # # marco architecture depth search based on micro architecture
    macro_architecture_config = {"learning_rate": mic_operation_weight_learning_rate,
                                 "weight_decay": mic_operation_weight_weight_decay,
                                 "input_dimension": graph.node_dim,
                                 "hidden_dimension": hidden_dimension,
                                 "output_dimension": graph.label_num,
                                 "micro_architecture": micro_architecture,
                                 "training_epoch": mac_archi_depth_explore_train_epoch}

    macro_architecture_depth = MarcoArchitectureDepthSearch(depth=mac_archi_depth_explore,
                                                            macro_architecture_config=macro_architecture_config,
                                                            device=device,
                                                            graph=graph)

    best_depth = macro_architecture_depth.macro_architecture_depth_search(feedback=feedback_mac_depth_evaluation)

    num_state_node = best_depth

    # macro architecture skip connection search
    pre_post_mlp_config = {"pre_mlp_input_dim": graph.node_dim,
                           "pre_mlp_output_dim": hidden_dimension,
                           "post_mlp_input_dim": hidden_dimension,
                           "post_mlp_output_dim": graph.label_num}

    identity_op_config = {"identity_op_dim": hidden_dimension,
                          "identity_op_micro_architecture": micro_architecture}

    archi_param_optim_config = {"archi_param_learning_rate": mac_archi_param_learning_rate,
                                "archi_param_weight_decay":  mac_archi_param_weight_decay,
                                "archi_param_temperature": mac_archi_param_temperature}

    identity_param_optim_config = {"identity_param_learning_rate": mic_operation_weight_learning_rate,
                                   "identity_param_weight_decay": mic_archi_param_weight_decay}

    macro_search = MacroArchitectureSkipConnectionSearch(num_state_node,
                                                         archi_param_optim_config,
                                                         identity_op_config,
                                                         identity_param_optim_config,
                                                         pre_post_mlp_config,
                                                         device)

    macro_architecture_list = macro_search.search(graph,
                                                  mac_archi_skip_connection_search_epoch,
                                                  mac_promising_skip_connection_archi_num)

    architecture_config = {"input_dimension": graph.node_dim,
                           "hidden_dimension": hidden_dimension,
                           "output_dimension": graph.label_num,
                           "learning_rate": mic_operation_weight_learning_rate,
                           "weight_decay": mic_operation_weight_weight_decay}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for macro_architecture in macro_architecture_list:

        if macro_architecture != []:
            model_test = MicroMacroArchitectureModelTrainTest(architecture_config=architecture_config,
                                                              macro_architecture=macro_architecture,
                                                              micro_architecture=micro_architecture,
                                                              device=device)

            model_test.train_test(graph, final_architecture_training_epoch)
            print("mac archi:", macro_architecture)
            print("mac depth:", len(macro_architecture)+1)
            print("mic archi", micro_architecture)
        else:
            print("Macro Architecture is []")