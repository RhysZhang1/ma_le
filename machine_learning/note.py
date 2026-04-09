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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
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
    ax2.scatter(slope, intercept, j_value, color='red', s=100, marker='o', label='当前拟合点')

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
    ax3.scatter(slope, intercept, color='red', s=100, marker='o', label='当前拟合点')

    ax3.set_xlabel('斜率 (w)')
    ax3.set_ylabel('截距 (b)')
    ax3.set_title('J(w,b) 二维等高线')
    ax3.legend()

    plt.tight_layout()
    plt.show()

#梯度下降：在当前位置走一小步，朝下降最快的方向
#局部最小值，不一定是全局最小值
#x_new=x_old−α∇f(x_old)

if __name__ == '__main__':
    xxhg()