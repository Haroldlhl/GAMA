import json
from typing import Dict, Tuple
from node import Node, Edge


def parse_graph_json(json_file_path: str) -> Tuple[Dict[str, Node], Dict[str, Edge]]:
    """
    解析包含节点和边信息的JSON文件
    
    参数:
        json_file_path: JSON文件的路径
        
    返回:
        一个元组，包含两个字典:
        - 第一个字典包含所有节点，键为节点ID，值为Node对象
        - 第二个字典包含所有边，键为边ID，值为Edge对象
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # 解析节点
        nodes = {}
        for node_id, node_data in data.get('nodes', {}).items():
            # 注意：修复了原数据中的拼写错误 "neigbors" 为 "neighbors"
            node = Node(
                id=node_data['id'],
                type=node_data['type'],
                area=node_data['area'],
                floor=node_data['floor'],
                unsearched_area=node_data['unsearched_area'],
                position=tuple(node_data['position']),  
                degree=node_data['degree'],
                estimated_finish_time=node_data['estimated_finish_time'],
                difficulty_factor=node_data['difficulty_factor'],
                allowed_uav_number=node_data['allowed_uav_number'],
                searching_uav_number=node_data['searching_uav_number']
            )
            nodes[node_id] = node
        
        # 解析边并更新节点的邻居信息
        edges = {}
        for edge_id, edge_data in data.get('edges', {}).items():
            edge = Edge(
                id=edge_data['id'],
                source=edge_data['source'],
                target=edge_data['target'],
                length=edge_data['length'],
                weight=edge_data['weight'],
                time_cost=edge_data['time_cost']
            )
            edges[edge_id] = edge
        return nodes, edges
    
    except FileNotFoundError:
        raise Exception(f"文件未找到: {json_file_path}")
    except json.JSONDecodeError:
        raise Exception(f"JSON解析错误: {json_file_path}")
    except KeyError as e:
        raise Exception(f"JSON数据缺少必要的字段: {e}")
    except Exception as e:
        raise Exception(f"解析JSON文件时发生错误: {str(e)}")

# 使用示例
if __name__ == "__main__":
    try:
        nodes, edges = parse_graph_json('map.json')
            
    except Exception as e:
        print(f"错误: {e}")
    