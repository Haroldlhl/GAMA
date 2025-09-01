import torch
from graph_structure import create_sample_building_graph, convert_to_pyg_data
from graph_structure import BuildingGraph, SpaceNode, RoomType,SpaceEdge, ConnectionType
from gcn_model import create_gcn_model
from sklearn.metrics.pairwise import cosine_similarity

def test_gcn_encoder():
    """测试GCN编码器"""
    print("=== 测试GCN建筑图编码 ===")
    
    # 1. 创建建筑图
    building_graph = create_sample_building_graph()
    print(f"创建建筑图: {len(building_graph.nodes)} 个节点, {sum(len(edges) for edges in building_graph.edges.values())} 条边")
    
    # 2. 转换为PyG格式
    graph_data = convert_to_pyg_data(building_graph, flight_speed=0.5)
    print(f"节点特征维度: {graph_data.x.shape[1]}")
    print(f"边权重范围: {graph_data.edge_weight.min():.3f} - {graph_data.edge_weight.max():.3f}")
    
    # 3. 创建GCN模型
    model = create_gcn_model(input_dim=graph_data.x.shape[1])
    print(f"GCN模型创建成功: {sum(p.numel() for p in model.parameters())} 个参数")
    
    # 4. 进行编码
    model.eval()
    with torch.no_grad():
        embeddings = model(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
    
    print(f"编码完成: {embeddings.shape}")
    
    # 5. 分析结果
    analyze_embeddings(embeddings, building_graph)
    
    return embeddings, graph_data

def analyze_embeddings(embeddings, building_graph):
    """分析编码结果"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = cosine_similarity(embeddings.numpy())
    node_ids = list(building_graph.nodes.keys())
    
    print("\n=== 节点编码相似度分析 ===")
    print("节点ID:", node_ids)
    
    print("\n相似度矩阵:")
    for i, id_i in enumerate(node_ids):
        similarities = [f"{similarity_matrix[i, j]:.3f}" for j in range(len(node_ids))]
        print(f"  {id_i}: [{', '.join(similarities)}]")
    
    # 找出最相似和最不相似的节点对
    max_sim = -1
    min_sim = 2
    max_pair = None
    min_pair = None
    
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            sim = similarity_matrix[i, j]
            if sim > max_sim:
                max_sim = sim
                max_pair = (node_ids[i], node_ids[j])
            if sim < min_sim:
                min_sim = sim
                min_pair = (node_ids[i], node_ids[j])
    
    print(f"\n最相似节点对: {max_pair} (相似度: {max_sim:.3f})")
    print(f"最不相似节点对: {min_pair} (相似度: {min_sim:.3f})")

def test_different_graph_sizes():
    """测试不同规模的图"""
    print("\n=== 测试不同规模图的适应性 ===")
    
    # 测试小图
    small_graph = BuildingGraph()
    small_graph.add_node(SpaceNode("room1", 20.0, RoomType.NORMAL, 0.8, 1))
    small_graph.add_node(SpaceNode("room2", 25.0, RoomType.NORMAL, 0.8, 1))
    small_graph.add_edge(SpaceEdge("room1", "room2", 3.0, ConnectionType.DOOR, 1.2))
    
    small_data = convert_to_pyg_data(small_graph)
    model = create_gcn_model(input_dim=small_data.x.shape[1])
    
    with torch.no_grad():
        small_embeddings = model(small_data.x, small_data.edge_index, small_data.edge_weight)
    
    print(f"小图编码: {small_embeddings.shape}")
    
    # 测试大图（使用相同的模型）
    large_graph = create_sample_building_graph()  # 5个节点
    large_data = convert_to_pyg_data(large_graph)
    
    with torch.no_grad():
        large_embeddings = model(large_data.x, large_data.edge_index, large_data.edge_weight)
    
    print(f"大图编码: {large_embeddings.shape}")
    print("✓ 同一模型成功处理了不同规模的图")

if __name__ == "__main__":
    # 运行测试
    embeddings, graph_data = test_gcn_encoder()
    test_different_graph_sizes()
    
    print("\n=== 测试完成 ===")