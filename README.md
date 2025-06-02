### 使用方法

用 `ucb_bias.py` 代替原来的 `ucb_bias`（或者新开个文件改个名字和原来的区分）

将 `group_model.py` 放在 `script_for_experiments\benchmark_utils` 文件夹下。

将 `group_estimate_bias.py` 放在 `duel_bias\feedback\group_estimate_bias.py` 文件夹下。

实验方法基本和原来的 `demo.py` 一样，改动如下：

```python
 # ======= 构造模型参数 =======
    s = random_state.normal(0, 2.0, size=num_arms)  # 技能值
    g = random_state.integers(0, 2, size=num_arms)  # 每个臂是否属于偏好类别（g_i ∈ {0,1}）
    r = get_user_bias_beta(user_num, alpha=2.0, beta=1.0)  # 每个用户对 g=1 的臂的偏好 r_k

    mean_user_bias = r
    n_parallel = get_n_parallel()

    # ======= 算法配置 =======
    comparison_list = [
        (DoubleThompsonSamplingBias, {'name': 'DT-B(*)', 'comparative_user_bias': mean_user_bias, 'is_track_residual': True}),
        (BiasSensitiveUCB, {'name': 'BS-UN(*)', 'comparative_user_bias': mean_user_bias, 'is_track_residual': False}),
    ] 
    # track_residual 用到的东西还没写完，暂时设成 false
    # 使用新的 ucb_bias.py 中的 BiasSensitiveUCB

    algorithms, parameters_list = zip(*comparison_list)

#     # ======= 初始化自定义环境 =======
    env = PreferenceGroupModelBias(g=g, s=s, r=r, random_state=random_state)

    print_info_problem(env.preference_matrix.preferences, r)
```

如果需要记录额外信息，可以在 `benchmark.py` 的 `run_single_algorithm()` 中添加代码。
