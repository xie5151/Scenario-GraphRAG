import heapq
import math
import time
import numpy as np


class GridAStar:
    def __init__(self, x_min, x_max, y_min, y_max, resolution=0.1, robot_radius=0.1):
        """
        初始化A*栅格地图
        :param x_min, x_max, y_min, y_max: 场景世界坐标边界（单位：m，对应Gazebo场景尺寸）
        :param resolution: 栅格分辨率（单位：m/格）
        :param robot_radius: 机器人膨胀半径（单位：m）
        """
        self.resolution = resolution
        self.robot_radius = robot_radius

        # 世界坐标转栅格坐标的偏移量（将负坐标转为正整数栅格）
        self.x_offset = -x_min
        self.y_offset = -y_min

        # 计算栅格总尺寸
        self.grid_width = int((x_max - x_min) / resolution)
        self.grid_height = int((y_max - y_min) / resolution)

        # 初始化栅格地图：0=可通行，1=障碍物
        self.grid = [[0 for _ in range(self.grid_height)] for _ in range(self.grid_width)]

        # 八邻域搜索（路径更平滑）
        self.motions = [
            (1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))
        ]

    def world_to_grid(self, wx, wy):
        """世界坐标 → 栅格坐标（整数）"""
        gx = int((wx + self.x_offset) / self.resolution)
        gy = int((wy + self.y_offset) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        """栅格坐标 → 世界坐标（路径输出用）"""
        wx = gx * self.resolution - self.x_offset
        wy = gy * self.resolution - self.y_offset
        return wx, wy

    def add_rect_obstacle(self, cx, cy, width, height):
        """
        添加矩形障碍物（对应SDF里的墙/箱体）
        :param cx, cy: 障碍物中心世界坐标（对应SDF的<pose> x y）
        :param width, height: 障碍物x、y方向的总长度（对应SDF的<size> x y）
        """
        # 加上机器人膨胀半径
        half_w = width / 2 + self.robot_radius
        half_h = height / 2 + self.robot_radius

        # 计算障碍物在栅格中的边界
        x_min_g, y_min_g = self.world_to_grid(cx - half_w, cy - half_h)
        x_max_g, y_max_g = self.world_to_grid(cx + half_w, cy + half_h)

        # 标记障碍区域
        for x in range(max(0, x_min_g), min(self.grid_width, x_max_g + 1)):
            for y in range(max(0, y_min_g), min(self.grid_height, y_max_g + 1)):
                self.grid[x][y] = 1

    def heuristic(self, x1, y1, x2, y2):
        """启发函数：欧氏距离"""
        return math.hypot(x2 - x1, y2 - y1)

    def plan(self, start_world, goal_world):
        """
        执行路径规划
        :param start_world: 起点世界坐标 (x, y)
        :param goal_world: 终点世界坐标 (x, y)
        :return: 世界坐标下的路径点列表 [(x1,y1), (x2,y2), ...]
        """
        start_x, start_y = self.world_to_grid(*start_world)
        goal_x, goal_y = self.world_to_grid(*goal_world)

        # 边界校验
        if self.grid[start_x][start_y] == 1 or self.grid[goal_x][goal_y] == 1:
            return None

        # 开放列表：(f值, x, y)，用堆实现优先队列
        open_heap = []
        heapq.heappush(open_heap, (0, start_x, start_y))

        # 记录父节点与代价
        came_from = {}
        g_cost = {(start_x, start_y): 0}
        closed = set()

        while open_heap:
            f_current, x, y = heapq.heappop(open_heap)

            # 到达终点
            if x == goal_x and y == goal_y:
                # 回溯路径
                path = []
                while (x, y) in came_from:
                    path.append(self.grid_to_world(x, y))
                    x, y = came_from[(x, y)]
                path.append(start_world)
                return path[::-1]  # 反转，起点→终点

            closed.add((x, y))

            # 遍历八邻域
            for dx, dy, cost in self.motions:
                nx, ny = x + dx, y + dy
                # 越界或障碍跳过
                if nx < 0 or nx >= self.grid_width or ny < 0 or ny >= self.grid_height:
                    continue
                if self.grid[nx][ny] == 1 or (nx, ny) in closed:
                    continue

                new_g = g_cost[(x, y)] + cost
                # 新路径代价更小，更新
                if (nx, ny) not in g_cost or new_g < g_cost[(nx, ny)]:
                    g_cost[(nx, ny)] = new_g
                    f = new_g + self.heuristic(nx, ny, goal_x, goal_y)
                    heapq.heappush(open_heap, (f, nx, ny))
                    came_from[(nx, ny)] = (x, y)

        # 搜索失败（无可行路径）
        return None


# ========== 实验执行入口（场景一：正方形围墙） ==========
if __name__ == "__main__":
    # --------------------------
    # 1. 场景初始化 + 人工配置障碍物（从SDF提取的真实参数）
    # --------------------------
    planner = GridAStar(
        x_min=-2.5, x_max=2.5,
        y_min=-2.5, y_max=2.5,
        resolution=0.1,
        robot_radius=0.1
    )

    # 场景一：4面围墙，参数完全对应SDF文件
    planner.add_rect_obstacle(cx=0.0, cy=1.925, width=4.0, height=0.15)   # 上墙 Wall_0
    planner.add_rect_obstacle(cx=-1.925, cy=0.0, width=0.15, height=4.0) # 左墙 Wall_2
    planner.add_rect_obstacle(cx=0.0, cy=-1.925, width=4.0, height=0.15) # 下墙 Wall_3
    planner.add_rect_obstacle(cx=1.925, cy=0.0, width=0.15, height=4.0)  # 右墙 Wall_4

    # 起点、终点（请替换为你LLM实验中完全一致的坐标，保证对比公平）
    start_point = (-1.0, -1.0)
    goal_point = (0.5, 0.0)

    # --------------------------
    # 2. 重复运行5次，统计纯规划耗时
    # --------------------------
    run_times = 5
    time_list = []
    final_path = None

    print(f"=== 单次耗时明细（共 {run_times} 次）===")
    for i in range(run_times):
        t0 = time.perf_counter()
        path = planner.plan(start_point, goal_point)
        t1 = time.perf_counter()
        single_time = t1 - t0
        time_list.append(single_time)
        print(f"第 {i+1} 次：{single_time:.4f} s  |  {single_time * 1000:.2f} ms")

        # 仅保存第一次的路径用于结果校验
        if path and final_path is None:
            final_path = path

    # --------------------------
    # 3. 统计量计算（均值、最小值、标准差）
    # --------------------------
    avg_time = np.mean(time_list)
    min_time = np.min(time_list)
    std_time = np.std(time_list)

    print("\n=== 统计结果 ===")
    if final_path:
        print(f"✅ 规划成功，路径点总数：{len(final_path)}")
        print(f"路径起点：{final_path[0]}")
        print(f"路径终点：{final_path[-1]}")
    else:
        print("❌ 无可行路径")
    print(f"平均规划耗时：{avg_time:.4f} s  |  {avg_time * 1000:.2f} ms")
    print(f"最短规划耗时：{min_time:.4f} s  |  {min_time * 1000:.2f} ms")
    print(f"耗时标准差：{std_time:.6f} s  |  {std_time * 1000:.4f} ms")
