# GAMA 训练的大纲

from config_parse import Config
from environment import Environment
from nets import ActorNetwork, CriticNetwork
from drone import Drone
from node import WorldGraph

def main():
    config_path = "config.json"
    # 1. 加载配置
    config = Config()
    # 创建map
    map_graph = WorldGraph()
    # 创建无人机
    drones = []
    for i in range(config.num_drones):
        drones.append(Drone())
    # 创建环境
    environment = Environment(map_graph, drones, config)

    # 创建网络
    actor_network = ActorNetwork(config)
    critic_network = CriticNetwork(config)
    graph_encoder = GNNEncoder(config)
    agent_encoder = AgentEncoder(config)

    # 训练
    train(config, environment, actor_network, critic_network, graph_encoder, agent_encoder)

def train(config, environment, actor_network, critic_network, graph_encoder, agent_encoder):
    # 获取环境