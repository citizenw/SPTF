import torch
from torch import nn
from .mlp import MultiLayerPerceptron, GraphMLP
from .tcn import TCN

device = 'cuda'


class SPTF(nn.Module):

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.if_mix = model_args["if_mix"]
        self.temp_dim_mix = model_args["temp_dim_mix"]
        self.if_anomaly = model_args["if_anomaly"]
        self.temp_dim_anomaly = model_args["temp_dim_anomaly"]
        self.if_time_norm = model_args["if_time_norm"]
        self.adj_mx = model_args["adj_mx"]

        self.hidden_dim_1 = self.embed_dim + self.temp_dim_anomaly * self.if_anomaly + \
                            self.temp_dim_mix * self.if_mix + \
                            self.node_dim * self.if_spatial + \
                            self.temp_dim_tid * self.if_time_in_day + \
                            self.temp_dim_diw * self.if_day_in_week
        self.hidden_dim_2 = self.embed_dim + self.output_len + \
                            self.temp_dim_anomaly * self.if_anomaly + \
                            self.temp_dim_mix * self.if_mix + \
                            self.node_dim * self.if_spatial + \
                            self.temp_dim_tid * self.if_time_in_day + \
                            self.temp_dim_diw * self.if_day_in_week

        self.adj_mx_encoder = nn.Sequential(
            *[GraphMLP(self.num_nodes, self.node_dim) for _ in range(1)],
            *[MultiLayerPerceptron(self.node_dim, self.node_dim) for _ in range(1)],
        )

        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.graph_model_1 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim_1, self.hidden_dim_1) for _ in range(1)],
        )
        self.graph_model_2 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim_2, self.hidden_dim_2) for _ in range(1)],
        )

        self.regression_layer_1 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim_1, self.hidden_dim_1) for _ in range(2)],
            nn.Linear(in_features=self.hidden_dim_1, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.output_len, bias=True),
        )

        self.regression_layer_2 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim_2, self.hidden_dim_2) for _ in range(2)],
            nn.Linear(in_features=self.hidden_dim_2, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.output_len, bias=True),
        )

        self.regression_layer = nn.Sequential(
            *[MultiLayerPerceptron(2*self.output_len, 2*self.output_len) for _ in range(3)],
            nn.Linear(in_features=2*self.output_len, out_features=self.output_len, bias=True),
        )

        self.tcn = TCN(input_size=self.input_len, output_size=self.embed_dim, n_layers=3, n_units=self.input_len,
                       dropout=0.1)

        if self.if_anomaly:
            self.anomaly_type = [0, 0.8, 0.9, 1.0, 1.1, 1.2, 1.8, 1.9, 2.0, 2.1, 2.2]

            self.anomaly_alpha = nn.Parameter(
                torch.empty(len(self.anomaly_type), 1))
            nn.init.uniform_(self.anomaly_alpha)

            self.anomaly_emb_layer = nn.Sequential(
                nn.Linear(in_features=self.input_len, out_features=self.temp_dim_anomaly, bias=True))
            self.anomaly_fusion_layer = nn.Sequential(
                *[MultiLayerPerceptron(self.temp_dim_anomaly + self.embed_dim + self.node_dim,
                                       self.temp_dim_anomaly + self.embed_dim + self.node_dim) for _ in range(3)],
                nn.Linear(in_features=self.temp_dim_anomaly + self.embed_dim + self.node_dim,
                          out_features=self.temp_dim_anomaly, bias=True),
            )

        if self.if_mix:
            self.mix_emb_layer = nn.Sequential(
                nn.Linear(in_features=self.input_len, out_features=self.temp_dim_mix, bias=True))
            self.mix_fusion_layer = nn.Sequential(
                *[MultiLayerPerceptron(self.temp_dim_mix + self.embed_dim + self.node_dim,
                                       self.temp_dim_mix + self.embed_dim + self.node_dim) for _ in range(3)],
                nn.Linear(in_features=self.temp_dim_mix + self.embed_dim + self.node_dim,
                          out_features=self.temp_dim_mix, bias=True)
            )



    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:

        traffic_flow_data = history_data[..., range(self.input_dim)]

        batch_size, _, num_nodes, _ = traffic_flow_data.shape
        traffic_flow_data = traffic_flow_data.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        traffic_flow_emb = self.tcn(traffic_flow_data.transpose(1, 2)).transpose(1, 2)

        node_emb = self.adj_mx[0].to(device)
        if self.if_spatial:
            node_emb = self.adj_mx_encoder(node_emb.unsqueeze(0)).expand(batch_size, -1, -1)

        mix_fusion_emb = []
        if self.if_mix:
            mix_data = history_data[..., 6].transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
            mix_emb = self.mix_emb_layer(mix_data)
            mix_fusion_emb.append(self.mix_fusion_layer(torch.cat([traffic_flow_emb] + [mix_emb] +
                                                                   [node_emb], dim=2)))

        anomaly_fusion_emb = []
        if self.if_anomaly:
            anomaly_data = history_data[..., 4].transpose(1, 2)
            for i in range(len(self.anomaly_type)):
                anomaly_data = torch.where(anomaly_data == self.anomaly_type[i], self.anomaly_alpha[i], anomaly_data)

            anomaly_emb = self.anomaly_emb_layer(anomaly_data)
            anomaly_fusion_emb.append(self.anomaly_fusion_layer(torch.cat([traffic_flow_emb] + [anomaly_emb] +
                                                                           [node_emb], dim=2)))

        tem_emb = []
        if self.if_time_in_day:
            if self.if_time_norm:
                t_i_d_data = history_data[..., 1] * self.time_of_day_size
            else:
                t_i_d_data = history_data[..., -1]
            tem_emb.append(self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)])
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            tem_emb.append(self.day_in_week_emb[(d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)])

        hidden_1 = torch.cat([traffic_flow_emb] + anomaly_fusion_emb + mix_fusion_emb + [node_emb] + tem_emb, dim=2)

        res_1 = self.graph_model_1(hidden_1)

        prediction_1 = self.regression_layer_1(res_1)

        hidden_2 = torch.cat([traffic_flow_emb] + anomaly_fusion_emb + mix_fusion_emb + [prediction_1] +
                             [node_emb] + tem_emb, dim=2)

        res_2 = self.graph_model_2(hidden_2)

        prediction_2 = self.regression_layer_2(res_2)

        prediction = [prediction_1, prediction_2]
        prediction = self.regression_layer(torch.cat(prediction, dim=2))

        return prediction.transpose(1, 2).unsqueeze(-1)
