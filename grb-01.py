import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("basic_lp")

# 添加变量（默认是连续变量）
x = model.addVar(name="x")
y = model.addVar(name="y")

# 设置目标函数：maximize 3x + 2y
model.setObjective(3 * x + 2 * y, GRB.MAXIMIZE)

# 添加约束
model.addConstr(x + 2 * y <= 4, "c1")
model.addConstr(3 * x + y <= 6, "c2")

# 求解
model.optimize()

# 打印结果
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found: x = {x.X}, y = {y.X}")
    print(f"Objective value = {model.ObjVal}")
else:
    print("No optimal solution found.")
