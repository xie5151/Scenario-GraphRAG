import heapq
import math
import time
import numpy as np


class GridAStar:
    def __init__(self, x_min, x_max, y_min, y_max, resolution=0.1, robot_radius=0.1):
        """
        初始化A*栅格地图
        :param x_min, x_max, y_min, y_max: 场景世界坐标边界（单位：m）
        :param resolution: 栅格分辨率（单位：m/格）
        :param robot_radius: 机器人膨胀半径（单位：m）
        """
        self.resolution = resolution
        self.robot_radius = robot_radius

        self.x_offset = -x_min
        self.y_offset = -y_min

        self.grid_width = int((x_max - x_min) / resolution)
        self.grid_height = int((y_max - y_min) / resolution)

        # 0=可通行，1=障碍物
        self.grid = [[0 for _ in range(self.grid_height)] for _ in range(self.grid_width)]

        # 八邻域搜索
        self.motions = [
            (1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))
        ]

    def world_to_grid(self, wx, wy):
        """世界坐标 → 栅格坐标"""
        gx = int((wx + self.x_offset) / self.resolution)
        gy = int((wy + self.y_offset) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        """栅格坐标 → 世界坐标"""
        wx = gx * self.resolution - self.x_offset
        wy = gy * self.resolution - self.y_offset
        return wx, wy

    def add_rect_obstacle(self, cx, cy, width, height):
        """添加矩形障碍物（对应墙体）"""
        half_w = width / 2 + self.robot_radius
        half_h = height / 2 + self.robot_radius

        x_min_g, y_min_g = self.world_to_grid(cx - half_w, cy - half_h)
        x_max_g, y_max_g = self.world_to_grid(cx + half_w, cy + half_h)

        for x in range(max(0, x_min_g), min(self.grid_width, x_max_g + 1)):
            for y in range(max(0, y_min_g), min(self.grid_height, y_max_g + 1)):
                self.grid[x][y] = 1

    def add_circle_obstacle(self, cx, cy, radius):
        """添加圆形/圆柱形障碍物"""
        total_radius = radius + self.robot_radius

        x_min_g, y_min_g = self.world_to_grid(cx - total_radius, cy - total_radius)
        x_max_g, y_max_g = self.world_to_grid(cx + total_radius, cy + total_radius)

        for x in range(max(0, x_min_g), min(self.grid_width, x_max_g + 1)):
            for y in range(max(0, y_min_g), min(self.grid_height, y_max_g + 1)):
                wx, wy = self.grid_to_world(x, y)
                dist = math.hypot(wx - cx, wy - cy)
                if dist <= total_radius:
                    self.grid[x][y] = 1

    def heuristic(self, x1, y1, x2, y2):
        """启发函数：欧氏距离"""
        return math.hypot(x2 - x1, y2 - y1)

    def plan(self, start_world, goal_world):
        """执行路径规划，返回世界坐标路径点列表"""
        start_x, start_y = self.world_to_grid(*start_world)
        goal_x, goal_y = self.world_to_grid(*goal_world)

        if self.grid[start_x][start_y] == 1 or self.grid[goal_x][goal_y] == 1:
            return None

        open_heap = []
        heapq.heappush(open_heap, (0, start_x, start_y))

        came_from = {}
        g_cost = {(start_x, start_y): 0}
        closed = set()

        while open_heap:
            f_current, x, y = heapq.heappop(open_heap)

            if x == goal_x and y == goal_y:
                path = []
                while (x, y) in came_from:
                    path.append(self.grid_to_world(x, y))
                    x, y = came_from[(x, y)]
                path.append(start_world)
                return path[::-1]

            closed.add((x, y))

            for dx, dy, cost in self.motions:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= self.grid_width or ny < 0 or ny >= self.grid_height:
                    continue
                if self.grid[nx][ny] == 1 or (nx, ny) in closed:
                    continue

                new_g = g_cost[(x, y)] + cost
                if (nx, ny) not in g_cost or new_g < g_cost[(nx, ny)]:
                    g_cost[(nx, ny)] = new_g
                    f = new_g + self.heuristic(nx, ny, goal_x, goal_y)
                    heapq.heappush(open_heap, (f, nx, ny))
                    came_from[(nx, ny)] = (x, y)

        return None


# ========== 实验执行入口（场景四：迷宫复杂场景） ==========
if __name__ == "__main__":
    # 1. 场景初始化
    planner = GridAStar(
        x_min=-3.0, x_max=3.0,
        y_min=-3.0, y_max=3.0,
        resolution=0.1,
        robot_radius=0.1
    )

    # 2. 添加外围4面主墙
    planner.add_rect_obstacle(cx=-2.425, cy=0.0, width=0.15, height=5.0)    # 左外墙 Wall_0
    planner.add_rect_obstacle(cx=0.0, cy=2.425, width=5.0, height=0.15)    # 上外墙 Wall_2
    planner.add_rect_obstacle(cx=2.425, cy=0.0, width=0.15, height=5.0)    # 右外墙 Wall_3
    planner.add_rect_obstacle(cx=0.0, cy=-2.425, width=5.0, height=0.15)   # 下外墙 Wall_4

    # 3. 添加内部8段短墙（迷宫结构）
    planner.add_rect_obstacle(cx=-1.937, cy=-1.467, width=1.0, height=0.15)  # Wall_11
    planner.add_rect_obstacle(cx=-0.22, cy=-1.866, width=0.15, height=1.0)   # Wall_13
    planner.add_rect_obstacle(cx=1.195, cy=-1.002, width=0.15, height=1.0)   # Wall_15
    planner.add_rect_obstacle(cx=1.288, cy=1.93, width=0.15, height=1.0)     # Wall_17
    planner.add_rect_obstacle(cx=1.91128, cy=0.4632, width=1.0, height=0.15) # Wall_19
    planner.add_rect_obstacle(cx=0.204, cy=0.215, width=0.15, height=1.0)    # Wall_21
    planner.add_rect_obstacle(cx=-1.064, cy=1.548, width=1.0, height=0.15)   # Wall_7
    planner.add_rect_obstacle(cx=-1.502, cy=0.092, width=0.15, height=1.0)   # Wall_9

    # 4. 添加2个圆柱形障碍物
    planner.add_circle_obstacle(cx=2.0, cy=2.0, radius=0.12)
    planner.add_circle_obstacle(cx=-2.0, cy=-2.0, radius=0.12)

    # 起点、终点 → 请替换为你LLM实验中完全一致的坐标
    start_point = (-2.0, -1.0)
    goal_point = (2.0, 1.0)

    # 5. 重复运行5次统计耗时
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

        if path and final_path is None:
            final_path = path

    # 6. 统计结果
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
