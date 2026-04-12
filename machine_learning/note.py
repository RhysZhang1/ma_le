import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#什么是机器学习：
# 机器学习是一种通过计算机程序自动学习和改进的领域。
# 它使计算机能够从数据中提取知识和模式，而无需明确指令。
#类型：
# 1.监督学习（更多）
# 2.非监督学习
# 3.强化学习
# 4.推荐系统

#监督学习：
# X -> Y  正确输入 -> 期望输出
#回归，分类
#回归：预测连续值
#分类：预测离散值
#无监督学习：
# 无标签数据，找到数据中的模式和结构
# 聚类：将相似的样本分组
# 聚类分析：发现数据中的隐藏模式
# 聚类分析的指标：
# 聚类的轮廓系数（Silhouette Coefficient）
# 聚类的平均距离（Average Distance）
# 聚类的内聚度（Intra-cluster Coherence）
# 聚类的分隔度（Inter-cluster Separation）
#other：异常检测，降维，特征提取

#线性回归
#平方误差代价函数：j(W,b) = (1/(2*m)) * np.sum((y_pred - y) ** 2)
#目的：让平方误差代价函数最小
def xxhg():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # ==================== 参数设置 ====================
    import os
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "huigui.xlsx")
    sheet_name = 0  # 工作表索引或名称，默认第一个表
    nrows = 81  # 读取前81行（包含表头）

    # ==================== 读取数据 ====================
    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=nrows)

    # 取第二列（索引1）作为x，第三列（索引2）作为y
    x_col = df.columns[1]  # 第二列的列名
    y_col = df.columns[2]  # 第三列的列名

    x = df[x_col].values.astype(float)  # 确保为数值类型
    y = df[y_col].values.astype(float)

    # ==================== 线性回归 ====================
    # 用 np.polyfit 拟合一次多项式，返回 [斜率, 截距]
    slope, intercept = np.polyfit(x, y, 1)

    # 生成回归直线的 y 值
    y_pred = slope * x + intercept

    # 计算均方误差的一半 J(w,b)
    m = len(x)
    j_value = (1/(2*m)) * np.sum((y_pred - y) ** 2)
    
    # 打印计算结果
    print("\n线性回归结果:")
    print(f"斜率 (slope): {slope:.4f}")
    print(f"截距 (intercept): {intercept:.4f}")
    print(f"J(w,b): {j_value:.4f}")

    # ==================== 绘图 ====================
    # 创建一个包含三个子图的图形，1行3列布局
    fig = plt.figure(figsize=(20, 6))

    # 第一个子图：线性回归散点图
    ax1 = fig.add_subplot(131)
    ax1.scatter(x, y, color='blue', label='原始数据', zorder=3)
    ax1.plot(x, y_pred, color='red', linewidth=2, label=f'回归直线: y = {slope:.3f}x + {intercept:.3f}')

    # 标注方程和 J 值
    equation_text = f'y = {slope:.4f}x + {intercept:.4f}\nJ(w,b) = {j_value:.4f}'
    ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_xlabel(f'第二列: {x_col}')
    ax1.set_ylabel(f'第三列: {y_col}')
    ax1.set_title('线性回归分析')
    # ax1.set_xlim(0, 80)  # 设置x轴范围为0到80
    # ax1.set_ylim(0, 80)  # 设置y轴范围为0到80
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 第二个子图：3D 图展示 J 值
    from mpl_toolkits.mplot3d import Axes3D
    ax2 = fig.add_subplot(132, projection='3d')

    # 生成斜率和截距的网格
    slope_range = np.linspace(slope - 0.1, slope + 0.1, 50)
    intercept_range = np.linspace(intercept - 5, intercept + 5, 50)
    slope_grid, intercept_grid = np.meshgrid(slope_range, intercept_range)

    # 计算每个斜率和截距组合的 J 值
    j_grid = np.zeros_like(slope_grid)
    for i in range(slope_grid.shape[0]):
        for j in range(slope_grid.shape[1]):
            y_pred_grid = slope_grid[i, j] * x + intercept_grid[i, j]
            j_grid[i, j] = (1/(2*m)) * np.sum((y_pred_grid - y) ** 2)

    # 绘制3D表面
    surf = ax2.plot_surface(slope_grid, intercept_grid, j_grid, cmap='viridis', alpha=0.8)
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)

    # 标记当前拟合的点
    ax2.scatter(slope, intercept, j_value, color='red', s=10, marker='o', label='当前拟合点')

    ax2.set_xlabel('斜率 (w)')
    ax2.set_ylabel('截距 (b)')
    ax2.set_zlabel('J(w,b)')
    ax2.set_title('J(w,b) 三维曲面')
    ax2.legend()

    # 第三个子图：二维等高线图展示 J 值
    ax3 = fig.add_subplot(133)
    
    # 绘制等高线（无背景填充）
    contour = ax3.contour(slope_grid, intercept_grid, j_grid, cmap='viridis', linewidths=1.0, levels=30)
    fig.colorbar(contour, ax=ax3, shrink=0.5, aspect=5)
    
    # 标记当前拟合的点
    ax3.scatter(slope, intercept, color='red', s=10, marker='o', label='当前拟合点')

    ax3.set_xlabel('斜率 (w)')
    ax3.set_ylabel('截距 (b)')
    ax3.set_title('J(w,b) 二维等高线')
    ax3.legend()

    plt.tight_layout()
    plt.show()

#梯度下降：在当前位置走一小步，朝下降最快的方向
#局部最小值，不一定是全局最小值
#x_new=x_old−α∇f(x_old)

def tdxj():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # ==================== 参数设置 ====================
    import os
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "huigui.xlsx")
    sheet_name = 0  # 工作表索引或名称，默认第一个表
    nrows = 81  # 读取前81行（包含表头）

    # ==================== 读取数据 ====================
    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=nrows)

    # 取第二列（索引1）作为x，第三列（索引2）作为y
    x_col = df.columns[1]  # 第二列的列名
    y_col = df.columns[2]  # 第三列的列名

    x = df[x_col].values.astype(float)  # 确保为数值类型
    y = df[y_col].values.astype(float)

    # ==================== 线性回归（用于比较） ====================
    # 用 np.polyfit 拟合一次多项式，返回 [斜率, 截距]
    slope_opt, intercept_opt = np.polyfit(x, y, 1)
    y_pred_opt = slope_opt * x + intercept_opt
    m = len(x)
    j_value_opt = (1/(2*m)) * np.sum((y_pred_opt - y) ** 2)

    # ==================== 梯度下降 ====================
    # 初始值
    w = 0.54
    b = 6
    learning_rate = 0.00001  # 学习率，增大以加快收敛
    num_iterations = 3000000  # 迭代次数

    # 保存历史记录
    w_history = [w]
    b_history = [b]
    j_history = [(1/(2*m)) * np.sum(((w * x + b) - y) ** 2)]

    # 梯度下降迭代
    for i in range(num_iterations):
        y_pred = w * x + b
        dw = (1/m) * np.sum((y_pred - y) * x)
        db = (1/m) * np.sum(y_pred - y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # 保存记录
        w_history.append(w)
        b_history.append(b)
        j_history.append((1/(2*m)) * np.sum(((w * x + b) - y) ** 2))

    # 打印梯度下降结果
    print("\n梯度下降结果:")
    print(f"最终斜率 (w): {w:.4f}")
    print(f"最终截距 (b): {b:.4f}")
    print(f"最终 J(w,b): {j_history[-1]:.4f}")
    print(f"最优斜率 (w_opt): {slope_opt:.4f}")
    print(f"最优截距 (b_opt): {intercept_opt:.4f}")
    print(f"最优 J(w,b): {j_value_opt:.4f}")

    # ==================== 绘图 ====================
    # 创建2x2布局
    fig = plt.figure(figsize=(16, 12))

    # 第一个子图：线性回归散点图
    ax1 = fig.add_subplot(221)
    ax1.scatter(x, y, color='blue', label='原始数据', zorder=3)
    ax1.plot(x, y_pred_opt, color='red', linewidth=2, label=f'最优回归直线: y = {slope_opt:.3f}x + {intercept_opt:.3f}')
    
    # 梯度下降最终拟合的直线
    y_pred_gd = w * x + b
    ax1.plot(x, y_pred_gd, color='green', linewidth=2, linestyle='--', label=f'梯度下降直线: y = {w:.3f}x + {b:.3f}')

    equation_text = f'最优: y = {slope_opt:.4f}x + {intercept_opt:.4f}\n梯度下降: y = {w:.4f}x + {b:.4f}'
    ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_xlabel(f'第二列: {x_col}')
    ax1.set_ylabel(f'第三列: {y_col}')
    ax1.set_title('线性回归分析')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 第二个子图：梯度下降迭代步数与J值
    ax2 = fig.add_subplot(222)
    ax2.plot(range(len(j_history)), j_history, color='blue', linewidth=2)
    ax2.scatter([0], j_history[0], color='red', s=100, marker='o', label='初始值')
    ax2.scatter([len(j_history)-1], j_history[-1], color='green', s=100, marker='o', label='最终值')

    ax2.set_xlabel('迭代步数')
    ax2.set_ylabel('J(w,b)')
    ax2.set_title('梯度下降：迭代步数 vs J值')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 生成斜率和截距的网格（用于3D图和等高线图）
    # 同时结合最优解、最终值和完整的梯度下降历史记录来确定范围，确保覆盖整个路线
    # 斜率范围
    w_candidates = [slope_opt, w, min(w_history), max(w_history)]
    w_min = min(w_candidates) - 0.5 * (max(w_candidates) - min(w_candidates))
    w_max = max(w_candidates) + 0.5 * (max(w_candidates) - min(w_candidates))
    
    # 截距范围
    b_candidates = [intercept_opt, b, min(b_history), max(b_history)]
    b_min = min(b_candidates) - 0.5 * (max(b_candidates) - min(b_candidates))
    b_max = max(b_candidates) + 0.5 * (max(b_candidates) - min(b_candidates))
    
    slope_range = np.linspace(w_min, w_max, 50)
    intercept_range = np.linspace(b_min, b_max, 50)
    slope_grid, intercept_grid = np.meshgrid(slope_range, intercept_range)

    # 计算每个斜率和截距组合的 J 值
    j_grid = np.zeros_like(slope_grid)
    for i in range(slope_grid.shape[0]):
        for j in range(slope_grid.shape[1]):
            y_pred_grid = slope_grid[i, j] * x + intercept_grid[i, j]
            j_grid[i, j] = (1/(2*m)) * np.sum((y_pred_grid - y) ** 2)

    # 第三个子图：3D 图 + 梯度下降路线
    from mpl_toolkits.mplot3d import Axes3D
    ax3 = fig.add_subplot(223, projection='3d')

    # 绘制3D表面
    surf = ax3.plot_surface(slope_grid, intercept_grid, j_grid, cmap='viridis', alpha=0.6)
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)

    # 绘制梯度下降路线
    ax3.plot(w_history, b_history, j_history, color='red', linewidth=2, marker='.', markersize=5, label='梯度下降路线')
    
    # 标记初始点、最终点和最优解
    ax3.scatter(w_history[0], b_history[0], j_history[0], color='blue', s=100, marker='o', label='初始点')
    ax3.scatter(w_history[-1], b_history[-1], j_history[-1], color='green', s=100, marker='o', label='最终点')
    ax3.scatter(slope_opt, intercept_opt, j_value_opt, color='orange', s=100, marker='*', label='最优解')

    ax3.set_xlabel('斜率 (w)')
    ax3.set_ylabel('截距 (b)')
    ax3.set_zlabel('J(w,b)')
    ax3.set_title('J(w,b) 三维曲面 + 梯度下降路线')
    ax3.legend()

    # 第四个子图：二维等高线图 + 梯度下降路线
    ax4 = fig.add_subplot(224)
    
    # 绘制等高线
    contour = ax4.contour(slope_grid, intercept_grid, j_grid, cmap='viridis', linewidths=1.0, levels=30)
    fig.colorbar(contour, ax=ax4, shrink=0.5, aspect=5)
    
    # 绘制梯度下降路线
    ax4.plot(w_history, b_history, color='red', linewidth=2, marker='.', markersize=5, label='梯度下降路线')
    
    # 标记初始点、最终点和最优解
    ax4.scatter(w_history[0], b_history[0], color='blue', s=100, marker='o', label='初始点')
    ax4.scatter(w_history[-1], b_history[-1], color='green', s=100, marker='o', label='最终点')
    ax4.scatter(slope_opt, intercept_opt, color='orange', s=100, marker='*', label='最优解')

    ax4.set_xlabel('斜率 (w)')
    ax4.set_ylabel('截距 (b)')
    ax4.set_title('J(w,b) 二维等高线 + 梯度下降路线')
    ax4.legend()

    plt.tight_layout()
    plt.show()
#多元线性回归
#f_w,b(x)=w1x1+w2x2+...+wnxn+b
#w=[w1,w2,...,wn]^T
#x=[x1,x2,...,xn]^T
#f_w,b(x)=w*x+b
if __name__ == '__main__':
    tdxj()