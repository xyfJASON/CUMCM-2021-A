# 项目结构



## 检查数据与预处理

- `checkdata.py`：检查数据
- `preprocess.py`：数据预处理
- `constant.py`：题目常数
- `readdata.py`：内有若干读数据函数



## 第一问/第二问：求解

基本思路：优化套优化，外层优化 $p$，内层优化拉伸量 $\lambda_i$

外层优化两种选择：人工改变步长搜索；调用 `minimize`

内层优化两种选择：分区（分层） `minimize` ；近似后转二次规划 `cvxopt`，同时加强 clip 条件，可以人工加强，也可以指数下降式加强

- `solve.py`：外层优化1，人工改变步长求解 $p$
- `solve2.py`：外层优化2，使用 `scipy.optimize.minimize` 求解 $p$
- `solve_with_minimize`：内层优化1，分区 `minimize`
- `solve_with_qp.py`：内层优化2，近似后转二次规划，并指数下降式加强 clip 条件
- `solve_with_qp_.py`：内层优化2，近似后转二次规划，并人工加强 clip 条件



## 第三问：计算接收比

- `calc_intersection.cpp`：计算三角形和圆的面积交
- `calc_intersection`：`cpp` 文件的编译结果，将被 `python` 调用
- `evaluate_in.txt`：`calc_intersection.cpp` 的输入，三角形坐标
- `evaluate.py`：计算接收比



## 结果

- `generate_result.py`：生成附件4对应的 csv 格式

- `result`

  - `quadratic`：内层优化采用二次规划的结果

    - `T1`

      `best_` 开头文件是外层优化采用 `minimize` 的结果；其余是外层优化人工变步长搜索的结果（搜索范围与步长在文件名中体现）

    - `T2`

      `best_` 开头文件是外层优化采用 `minimize` 的结果；其余是外层优化人工变步长搜索的结果（搜索范围与步长在文件名中体现）

  - `minimize`：内层优化采用分层优化的结果

    - `T1`
    
  - `result1.csv`：要提交的附件4的sheet1
  
  - `result2.csv`：要提交的附件4的sheet2
  
  - `result3.csv`：要提交的附件4的sheet3



## 其他

- `visualize.py`：可视化模块

