def min_path_sum(mat):
    if not mat or not mat[0]:
        return 0

    rows, cols = len(mat), len(mat[0])
    dp = [[0] * cols for _ in range(rows)]

    dp[0][0] = mat[0][0]

    # 初始化第一行
    for col in range(1, cols):
        dp[0][col] = dp[0][col - 1] + mat[0][col]

    # 初始化第一列
    for row in range(1, rows):
        dp[row][0] = dp[row - 1][0] + mat[row][0]

    # 填充动态规划表
    for row in range(1, rows):
        for col in range(1, cols):
            dp[row][col] = min(dp[row - 1][col], dp[row][col - 1]) + mat[row][col]

    return dp[-1][-1]


# 示例矩阵
mat = [
    [1, 2, 3],
    [4, 5, 6],
    [2, 1, 3]
]

# 输出结果
result = min_path_sum(mat)
print("最小路径和:", result)  # 输出: 11
