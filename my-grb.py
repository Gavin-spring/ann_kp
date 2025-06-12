import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("my_grb_model")

# 添加变量
x = model.addVar(name="x", vtype=GRB.INTEGER, lb=0, ub=10)  # 整数变量
y = model.addVar(name="y", vtype=GRB.CONTINUOUS, lb=0, ub=10)  # 连续变量

# 设置目标函数：maximize 2x + 3y
model.setObjective(2 * x + 3 * y, GRB.MAXIMIZE)

# 添加约束
model.addConstr(x + 2 * y <= 8, "c1")
model.addConstr(3 * x + y <= 12, "c2")
model.addConstr(x - y >= 1, "c3")

# 求解
model.optimize()

# 打印结果
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found: x = {x.X}, y = {y.X}")
    print(f"Objective value = {model.ObjVal}")
    
else:
    print("No optimal solution found.")
    
# 打印所有变量的值
for v in model.getVars():
    print(f"{v.VarName} = {v.X}")
    
# 打印所有约束的值
for c in model.getConstrs():
    print(f"{c.ConstrName}: {c.Slack} slack, {c.Pi} dual value")
    
# 打印模型信息
print(f"Model status: {model.status}")

# 打印模型的目标函数
print(f"Objective function: {model.getObjective().getValue()}")

# 打印模型的变量信息
print("Variables:") 
for v in model.getVars():
    print(f"  {v.VarName}: {v.X} (LB: {v.LB}, UB: {v.UB}, VType: {v.VType})")
    