import sys
from collections import defaultdict

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [0] * (2 * self.size)
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = max(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, pos, value):
        pos += self.size
        self.tree[pos] = value
        pos >>= 1
        while pos >= 1:
            new_val = max(self.tree[2 * pos], self.tree[2 * pos + 1])
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
            pos >>= 1

    def query(self, l, r):
        res = -float('inf')
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

def main():
    sys.setrecursionlimit(1 << 25)
    n, m = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    a = [0] + a  # 节点编号从1开始

    adj = defaultdict(list)
    parent = [0] * (n + 1)
    for _ in range(n - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # 确定父节点，构建树结构（以1为根）
    parent[1] = -1
    stack = [1]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v != parent[u]:
                parent[v] = u
                stack.append(v)

    # 计算每个节点的子节点
    children = defaultdict(list)
    for v in range(2, n + 1):
        u = parent[v]
        children[u].append(v)

    # 计算每个节点的权值
    weight = [0] * (n + 1)
    for u in range(1, n + 1):
        if not children[u]:  # 叶子节点
            weight[u] = a[u]
        else:
            max_xor = -1
            for v in children[u]:
                current_xor = a[u] ^ a[v]
                if current_xor > max_xor:
                    max_xor = current_xor
            weight[u] = max_xor

    # 生成DFS序，用于子树查询
    in_time = [0] * (n + 1)
    out_time = [0] * (n + 1)
    time = 0
    def dfs(u):
        nonlocal time
        time += 1
        in_time[u] = time
        for v in children[u]:
            dfs(v)
        out_time[u] = time
    dfs(1)

    # 初始化线段树（用于区间查询和子树查询）
    seg_data = [0] * (n + 1)
    for u in range(1, n + 1):
        seg_data[in_time[u]] = weight[u]
    seg_tree = SegmentTree(seg_data[1: n + 1])

    for _ in range(m):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            x = int(parts[1])
            y = int(parts[2])
            a[x] = y
            # 更新当前节点的权值
            if not children[x]:
                new_weight = y
            else:
                max_xor = -1
                for v in children[x]:
                    current_xor = y ^ a[v]
                    if current_xor > max_xor:
                        max_xor = current_xor
                new_weight = max_xor
            weight[x] = new_weight
            seg_tree.update(in_time[x] - 1, new_weight)
            # 更新祖先节点的权值
            u = parent[x]
            while u != -1:
                if not children[u]:
                    new_u_weight = a[u]
                else:
                    max_xor = -1
                    for v in children[u]:
                        current_xor = a[u] ^ a[v]
                        if current_xor > max_xor:
                            max_xor = current_xor
                    new_u_weight = max_xor
                if weight[u] != new_u_weight:
                    weight[u] = new_u_weight
                    seg_tree.update(in_time[u] - 1, new_u_weight)
                u = parent[u]
        elif parts[0] == '2':
            x = int(parts[1])
            y = int(parts[2])
            max_weight = -float('inf')
            for u in range(x, y + 1):
                if weight[u] > max_weight:
                    max_weight = weight[u]
            print(max_weight)
        elif parts[0] == '3':
            x = int(parts[1])
            l = in_time[x]
            r = out_time[x]
            max_weight = seg_tree.query(l - 1, r - 1)
            print(max_weight)

if __name__ == "__main__":
    main()