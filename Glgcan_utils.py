import numpy as np
import os
import torch
import json

class Glgcan():

    def __init__(self):

        current_path = os.path.abspath(__file__)
        path = os.path.abspath((os.path.dirname(current_path)))
        file = "/graph_data_Glgcan/SugarBase/"
        file_path = path + file
        file_name_list = os.listdir(path + file)

        X = []
        A = []
        Y = []
        data_num = 0
        label = []
        label_num = 2

        for data in file_name_list:
            data_num += 1
            with open(file_path + data, "r") as load_f:
                load_dict = json.load(load_f)
                X.append(np.array(load_dict["nodes_attribute"]))
                A.append(np.array(load_dict["adjacency_matrix"]))
                Y.append(np.array(load_dict["label"]))
                if label == []:
                    label.append(Y[0])
                else:
                    if np.array(load_dict["label"]) in label:
                        pass
                    else:
                        label.append(np.array(load_dict["label"]))
                        label_num += 1

        index = [i for i in range(data_num)]
        # random.shuffle(index)
        index_train = data_num - 200
        train_index = index[:index_train]
        val_index = index[index_train:index_train + 100]
        test_index = index[index_train + 100:]

        train_X = [X[i] for i in train_index]
        train_A = [A[i] for i in train_index]
        train_Y = [Y[i] for i in train_index]
        train_Y = np.array(train_Y)

        val_X = [X[i] for i in val_index]
        val_A = [A[i] for i in val_index]
        val_Y = [Y[i] for i in val_index]
        val_Y = np.array(val_Y)

        test_X = [X[i] for i in test_index]
        test_A = [A[i] for i in test_index]
        test_Y = [Y[i] for i in test_index]
        test_Y = np.array(test_Y)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # x to batch
        self.train_x, self.train_batch = self.X_batch_operator(train_X)
        self.val_x, self.val_batch = self.X_batch_operator(val_X)
        self.test_x, self.test_batch = self.X_batch_operator(test_X)

        # adj to list
        train_edge_index_list = self.edge_index_operator(train_A)
        val_edge_index_list = self.edge_index_operator(val_A)
        test_edge_index_list = self.edge_index_operator(test_A)

        # adj to batch
        self.train_edge_index = self.A_batch_operator(train_edge_index_list)
        self.val_edge_index = self.A_batch_operator(val_edge_index_list)
        self.test_edge_index = self.A_batch_operator(test_edge_index_list)

        # list to torch
        # self.train_y = torch.tensor(train_Y, dtype=torch.float32)[:, None]
        # self.val_y = torch.tensor(val_Y, dtype=torch.float32)[:, None]
        # self.test_y = torch.tensor(test_Y, dtype=torch.float32)[:, None]

        # self.train_y = self.Y_operator(self.train_Y)
        # self.val_y = self.Y_operator(self.val_Y)
        # self.test_y = self.Y_operator(self.test_Y)

        # for CrossEntropyLoss
        self.train_y = torch.tensor(train_Y, dtype=torch.int64)
        self.val_y = torch.tensor(val_Y, dtype=torch.int64)
        self.test_y = torch.tensor(test_Y, dtype=torch.int64)

        self.train_x = self.train_x.to(self.device)
        self.train_batch = self.train_batch.to(self.device)
        self.val_x = self.val_x.to(self.device)
        self.val_batch = self.val_batch.to(self.device)
        self.test_x = self.test_x.to(self.device)
        self.test_batch = self.test_batch.to(self.device)
        self.train_edge_index = self.train_edge_index.to(self.device)
        self.val_edge_index = self.val_edge_index.to(self.device)
        self.test_edge_index = self.test_edge_index.to(self.device)
        self.train_y = self.train_y.to(self.device)
        self.val_y = self.val_y.to(self.device)
        self.test_y = self.test_y.to(self.device)

        self.data_name = "Glgcan"
        self.label_num = 2
        self.node_dim = 73

    def A_batch_operator(self, edge_list):

        source_index = []
        target_index = []
        ii = 0
        offset = 0

        for edge in edge_list:
            source = edge[0].tolist()
            target = edge[1].tolist()

            if ii == 0:
                source_index = source
                target_index = target
                offset = source_index[-1] + 1
                ii += 1
            else:
                source_index = source_index + [i + offset for i in source]
                target_index = target_index + [i + offset for i in target]
                offset = source_index[-1] + 1

        batch_edge_index = torch.tensor([source_index, target_index], dtype=torch.int64)

        return batch_edge_index

    def X_batch_operator(self, x_list):

        i = 0
        X = 0
        batch_list = []

        for x in x_list:
            if i == 0:
              X = x
              batch_list = [0] * len(x)
            else:
              X = np.concatenate((X, x), axis=0)
              batch_list = batch_list + [i] * len(x)
            i += 1

        X = torch.tensor(X, dtype=torch.float32)
        batch_list = torch.tensor(batch_list, dtype=torch.int64)

        return X, batch_list

    def Y_operator(self, Y):

        Y_ = []
        for y in Y:
            if y.item() == 1:
                Y_.append([1, 0])
            else:
                Y_.append([0, 1])

        return torch.tensor(Y_, dtype=torch.float32)

    def edge_index_operator(self, adj):

        edge_index_list = []
        adj_p = []
        for index in range(len(adj)):
            adj_temp = adj[index]
            source_list = []
            target_list = []
            for j in range(adj_temp.shape[0]):
                for i in range(adj_temp.shape[0]):
                    if (adj_temp[i][j] != 0) and (i != j):
                        source_list.append(j)
                        target_list.append(i)
            if source_list == [] or target_list == []:
                adj_p.append(adj_temp)
                print(adj_temp)
            edge_index = [source_list, target_list]
            edge_index = torch.tensor(edge_index, dtype=torch.int64)
            edge_index_list.append(edge_index)
        return edge_index_list

if __name__=="__main__":

    a = Glgcan()
    c = 1