import gurobipy as gp
from gurobipy import GRB

# 数据
values = [10, 5, 15, 7, 6]     # 价值
weights = [2, 3, 5, 7, 1]      # 重量
capacity = 10                 # 背包容量
n = len(values)

# 创建模型
model = gp.Model("knapsack_01")

# 添加变量：x[i] = 1 表示选中第 i 个物品，变量类型为 BINARY（0 或 1）
x = model.addVars(n, vtype=GRB.BINARY, name="x")

# 设置目标函数：maximize 总价值
model.setObjective(gp.quicksum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

# 添加约束：总重量不能超过容量
model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "capacity")

# 求解
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最大价值 = {model.ObjVal}")
    print("选中的物品：")
    for i in range(n):
        if x[i].X > 0.5:
            print(f"  物品{i}：价值 = {values[i]}, 重量 = {weights[i]}")
else:
    print("没有找到最优解")
