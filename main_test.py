import torch
import numpy as np
from Glgcan_utils_cross_validation import cross_validation_graph
from architecture_build_test import MicroMacroArchitectureModelTrainTest


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_architecture_training_epoch = 100
    mic_operation_weight_learning_rate = 0.01
    mic_operation_weight_weight_decay = 0.0005
    hidden_dimension = 128

    micro_architecture = ['ARMAConv', 'BatchNorm', 'LeakyRelu']
    macro_architecture = [[1, 1, 0, 0, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1],
                          [1, 1, 0, 1, 1],
                          [1, 1, 0, 1],
                          [1, 0, 0],
                          [1, 0],
                          [1]]

    acc_list = []
    pre_list = []
    r_list = []
    f1_list = []
    epoch_list = []

    for graph in cross_validation_graph():
        architecture_config = {"input_dimension": graph.node_dim,
                               "hidden_dimension": hidden_dimension,
                               "output_dimension": graph.label_num,
                               "learning_rate": mic_operation_weight_learning_rate,
                               "weight_decay": mic_operation_weight_weight_decay}
        model_test = MicroMacroArchitectureModelTrainTest(architecture_config=architecture_config,
                                                          macro_architecture=macro_architecture,
                                                          micro_architecture=micro_architecture,
                                                          device=device)
        acc_test, p_test, r_test, f1_test, epoch = model_test.train_test(graph,
                                                                         final_architecture_training_epoch,
                                                                         manner="test_f1")
        acc_list.append(acc_test)
        pre_list.append(p_test)
        r_list.append(r_test)
        f1_list.append(f1_test)
        epoch_list.append(epoch)

    acc_array = np.array(acc_list)
    pre_array = np.array(pre_list)
    r_array = np.array(r_list)
    f1_array = np.array(f1_list)

    print("Average Accuracy", np.mean(acc_array), np.std(acc_array))
    print("Average Precision", np.mean(pre_array), np.std(pre_array))
    print("Average Recall", np.mean(r_array), np.std(r_array))
    print("Average F1-Score", np.mean(f1_array), np.std(f1_array))