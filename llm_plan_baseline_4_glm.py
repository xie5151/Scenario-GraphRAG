#!/usr/bin/env python3
import dashscope
import re
import math
import time
import rospy
import os
from openai import OpenAI
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle

# -------------------------- 核心配置 --------------------------
DASHSCOPE_API_KEY = "sk-fba66d331d824c36ae5ff30960c93aea"
dashscope.api_key = DASHSCOPE_API_KEY

START_POINT =(1.8,-1.0,0)
END_POINT =(-2.2,-1.8,90)

# 长墙(5m)：Wall0/Wall2/Wall3/Wall4 | 短墙(1m)：其余8根
WALLS = [
    # 长墙(5m)
    {"name": "Wall0", "x": -2.425, "y": 0, "length": 5, "thick": 0.15, "angle": math.pi/2},
    {"name": "Wall2", "x": 0, "y": 2.425, "length": 5, "thick": 0.15, "angle": 0},
    {"name": "Wall3", "x": 2.425, "y": 0, "length": 5, "thick": 0.15, "angle": -math.pi/2},
    {"name": "Wall4", "x": 0, "y": -2.425, "length": 5, "thick": 0.15, "angle": math.pi},
    # 短墙(1m)
    {"name": "Wall7", "x": -1.064, "y": 1.548, "length": 1, "thick": 0.15, "angle": 0},
    {"name": "Wall9", "x": -1.502, "y": 0.092, "length": 1, "thick": 0.15, "angle": -math.pi/2},
    {"name": "Wall11", "x": -1.937, "y": -1.467, "length": 1, "thick": 0.15, "angle": 0},
    {"name": "Wall13", "x": -0.22, "y": -1.866, "length": 1, "thick": 0.15, "angle": -math.pi/2},
    {"name": "Wall15", "x": 1.195, "y": -1.002, "length": 1, "thick": 0.15, "angle": math.pi/2},
    {"name": "Wall17", "x": 1.288, "y": 1.93, "length": 1, "thick": 0.15, "angle": -math.pi/2},
    {"name": "Wall19", "x": 1.91128, "y": 0.4632, "length": 1, "thick": 0.15, "angle": 0},
    {"name": "Wall21", "x": 0.204, "y": 0.215, "length": 1, "thick": 0.15, "angle": -math.pi/2},
]

# 圆柱障碍物
CYLINDERS = [
    {"name": "obstacle_1", "x": 2.0, "y": 2.0, "radius": 0.12},
    {"name": "obstacle_2", "x": -2.0, "y": -2.0, "radius": 0.12},
]

SAFE_DISTANCE = 0.05

MAX_LINEAR = 0.22
MAX_ANGULAR = 1.82
CONTROL_FREQUENCY = 20
START_WAIT_TIME = 3

# 全局里程计变量
current_x = START_POINT[0]
current_y = START_POINT[1]
current_theta = math.radians(START_POINT[2])

# -------------------------- 可视化：墙体+圆柱障碍+路径 --------------------------
def visualize_path(path_points):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # 画墙体：红色半透明
    for wall in WALLS:
        x, y = wall["x"], wall["y"]
        L, T = wall["length"], wall["thick"]
        angle = wall["angle"]
        # 计算矩形绘制坐标
        if abs(angle) in (0, math.pi):
            rx = x - L/2
            ry = y - T/2
            w, h = L, T
        else:
            rx = x - T/2
            ry = y - L/2
            w, h = T, L
        rect = patches.Rectangle((rx, ry), w, h, color='red', alpha=0.5, label='Walls' if wall["name"] == "Wall0" else "")
        ax.add_patch(rect)

    # 画圆柱障碍：蓝色实心圆
    for cyl in CYLINDERS:
        circle = Circle((cyl["x"], cyl["y"]), cyl["radius"], color='blue', label='Cylinder Obstacles' if cyl["name"] == "obstacle_1" else "")
        ax.add_patch(circle)

    # 画规划路径
    if path_points:
        x_coords = [p[0] for p in path_points]
        y_coords = [p[1] for p in path_points]
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Planned Path')
        ax.scatter(x_coords, y_coords, c='darkblue', s=15)

    # 起点（绿色）/终点（橙色）
    ax.scatter(START_POINT[0], START_POINT[1], c='green', s=150, marker='*', label=f'Start ({START_POINT[0]}, {START_POINT[1]})')
    ax.scatter(END_POINT[0], END_POINT[1], c='orange', s=150, marker='p', label=f'End ({END_POINT[0]}, {END_POINT[1]})')

    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('TurtleBot3 Plaza Scene - Path Planning', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.show(block=True)

# -------------------------- 碰撞检测：墙体+圆柱 --------------------------
def is_point_collide_wall(px, py, wall):
    """检测点是否与墙体碰撞"""
    wx, wy = wall["x"], wall["y"]
    L, T = wall["length"], wall["thick"]
    angle = wall["angle"]
    # 基于墙体旋转角度判断碰撞区域
    dx = abs(px - wx)
    dy = abs(py - wy)
    if abs(angle) in (0, math.pi):
        return dx <= (L/2 + SAFE_DISTANCE) and dy <= (T/2 + SAFE_DISTANCE)
    else:
        return dx <= (T/2 + SAFE_DISTANCE) and dy <= (L/2 + SAFE_DISTANCE)

def is_point_collide_cylinder(px, py, cyl):
    """检测点是否与圆柱障碍碰撞"""
    dx = px - cyl["x"]
    dy = py - cyl["y"]
    dist = math.hypot(dx, dy)
    return dist <= (cyl["radius"] + SAFE_DISTANCE)

def check_path_valid(path_points):
    """验证路径是否有效：无碰撞+起止点匹配"""
    if not path_points:
        rospy.logerr("路径点为空！")
        return False

    # 验证起止点匹配（容差0.2m）
    start_ok = math.isclose(path_points[0][0], START_POINT[0], abs_tol=0.2) and \
               math.isclose(path_points[0][1], START_POINT[1], abs_tol=0.2)
    end_ok = math.isclose(path_points[-1][0], END_POINT[0], abs_tol=0.2) and \
             math.isclose(path_points[-1][1], END_POINT[1], abs_tol=0.2)
    if not start_ok or not end_ok:
        rospy.logerr(f"起点/终点不匹配！规划起点：{path_points[0]}, 规划终点：{path_points[-1]}")
        return False

    # 检测墙体碰撞
    for (px, py) in path_points:
        for wall in WALLS:
            if is_point_collide_wall(px, py, wall):
                rospy.logerr(f"碰撞墙体{wall['name']}！点 ({px:.2f}, {py:.2f})")
                return False
        # 检测圆柱碰撞
        for cyl in CYLINDERS:
            if is_point_collide_cylinder(px, py, cyl):
                rospy.logerr(f"碰撞圆柱{cyl['name']}！点 ({px:.2f}, {py:.2f})")
                return False

    rospy.loginfo("路径验证通过：无墙体/圆柱碰撞！")
    return True

# -------------------------- 机器人复位：Gazebo初始姿态设置 --------------------------
def set_robot_initial_pose():
    rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
    try:
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        initial_state = ModelState()
        initial_state.model_name = 'turtlebot3_burger'

        # 位置复位
        initial_state.pose.position.x = START_POINT[0]
        initial_state.pose.position.y = START_POINT[1]
        initial_state.pose.position.z = 0.0

        # 朝向复位（四元数）
        yaw = math.radians(START_POINT[2])
        initial_state.pose.orientation.z = math.sin(yaw / 2)
        initial_state.pose.orientation.w = math.cos(yaw / 2)
        initial_state.reference_frame = 'world'

        resp = set_model_state(initial_state)
        if resp.success:
            rospy.loginfo(f"机器人复位成功！起点：({START_POINT[0]}, {START_POINT[1]})，朝向：{START_POINT[2]}°")
    except rospy.ServiceTimeoutException:
        rospy.logerr("连接/gazebo/set_model_state服务超时！请检查Gazebo是否启动")
    except Exception as e:
        rospy.logerr(f"机器人复位失败：{str(e)}")

# -------------------------- LLM路径规划--------------------------
def get_llm_path():
    prompt = f"""
你是TurtleBot3 Burger专业运动规划师，严格遵循要求，仅返回路径点组合，无任何其他文字/符号/说明！
场景：TurtleBot3 Plaza仿真场景，包含12段木质墙体+2个圆柱障碍物，
1. 外框：4面5m长墙围成正方形区域，墙厚0.15m，正方形中心在(0,0)，四边对应上下左右外框；
2. 内部：8面1m短墙随机分布在正方形内，无规则但不连通，核心障碍区在(0,0)到(1.5,1.5)之间，是规划重点避障区域；
3. 圆柱：obstacle_1在右上角(2.0,2.0)，obstacle_2在左下角(-2.0,-2.0)
精准参数如下：
==================== 墙体参数（共12段，厚度均0.15m，高0.5m,旋转角度为与X轴夹角） ====================
1. Wall0：坐标(-2.425, 0)，长度5m，旋转90°（垂直）
2. Wall2：坐标(0, 2.425)，长度5m，旋转0°（水平）
3. Wall3：坐标(2.425, 0)，长度5m，旋转-90°（垂直）
4. Wall4：坐标(0, -2.425)，长度5m，旋转180°（水平）
5. Wall7：坐标(-1.064, 1.548)，长度1m，旋转0°（水平）
6. Wall9：坐标(-1.502, 0.092)，长度1m，旋转-90°（垂直）
7. Wall11：坐标(-1.937, -1.467)，长度1m，旋转0°（水平）
8. Wall13：坐标(-0.22, -1.866)，长度1m，旋转-90°（垂直）
9. Wall15：坐标(1.195, -1.002)，长度1m，旋转90°（垂直）
10. Wall17：坐标(1.288, 1.93)，长度1m，旋转-90°（垂直）
11. Wall19：坐标(1.91128, 0.4632)，长度1m，旋转0°（水平）
12. Wall21：坐标(0.204, 0.215)，长度1m，旋转-90°（垂直）
==================== 圆柱障碍物参数（共2个，高0.25m） ====================
1. obstacle_1：中心坐标(2.0, 2.0)，半径0.12m
2. obstacle_2：中心坐标(-2.0, -2.0)，半径0.12m

任务：规划从起点A(1.8,-1.0,0) 到终点B(-2.2,-1.8,90)的平滑避障路径
核心要求：
1. 路径点必须避开所有墙体和圆柱，无任何碰撞，严格遵守安全距离0.2m
2. 路径点步长最大0.1米，保证机器人运动平滑
3. 优先级：优先避障 → 其次到达终点 → 最后路径平滑性
4. 输出格式严格遵循：(x1,y1) → (x2,y2) → ... → (xn,yn)
"""
    try:
        client = OpenAI(
            api_key= dashscope.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        response = client.chat.completions.create(
            model="glm-5.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            extra_body={"enable_thinking": True}
        )
        raw = response.choices[0].message.content.strip()
        rospy.loginfo(f"glm-5.1 返回路径：{raw}")

        usage = response.usage
        rospy.loginfo(f"单次调用Token明细：输入={usage.prompt_tokens}, 输出={usage.completion_tokens}, 总计={usage.total_tokens}")
        
        return raw, usage
    except Exception as e:
        rospy.logerr(f"glm-5.1 调用错误：{e}")
        return None, None

# -------------------------- 路径解析与平滑 --------------------------
def parse_path(raw_path):
    """解析LLM返回的路径点为浮点数坐标列表"""
    if not raw_path:
        return None
    # 正则匹配(x,y)格式，支持正负浮点数
    pattern = re.compile(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)')
    points = pattern.findall(raw_path)
    if not points:
        rospy.logerr("未解析到有效路径点！请检查LLM输出格式")
        return None
    # 转换为浮点数
    path_points = [(float(x), float(y)) for x, y in points]
    rospy.loginfo(f"解析出{len(path_points)}个原始路径点")
    return path_points

def interpolate_path(path_points, step=0.1):
    """路径插值平滑：保证步长不超过0.1m"""
    if len(path_points) < 2:
        return path_points
    smooth_path = [path_points[0]]
    for i in range(1, len(path_points)):
        x0, y0 = path_points[i-1]
        x1, y1 = path_points[i]
        # 计算两点间距离
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1e-6:
            continue
        # 计算插值点数
        n = max(1, int(dist / step))
        # 线性插值
        for j in range(1, n+1):
            factor = j / n
            x = x0 + factor * (x1 - x0)
            y = y0 + factor * (y1 - y0)
            smooth_path.append((round(x, 3), round(y, 3)))
    rospy.loginfo(f"插值后得到{len(smooth_path)}个平滑路径点")
    rospy.loginfo("\n========== 平滑后的最终路径点 ==========")
    for i, p in enumerate(smooth_path):
        rospy.loginfo(f"点 {i}: ({p[0]:.3f}, {p[1]:.3f})")
    return smooth_path

# -------------------------- 里程计回调：获取机器人实时位姿 --------------------------
def odom_callback(msg):
    global current_x, current_y, current_theta
    # 实时位置
    current_x = msg.pose.pose.position.x
    current_y = msg.pose.pose.position.y
    # 实时朝向（四元数转欧拉角yaw）
    q = msg.pose.pose.orientation
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y ** 2 + q.z ** 2)
    current_theta = math.atan2(siny_cosp, cosy_cosp)

# -------------------------- 闭环运动控制：跟踪路径点 --------------------------
def path_to_gazebo_vel(path_points):
    """将规划路径转换为速度指令，控制机器人在Gazebo中运动"""
    global current_x, current_y, current_theta
    # 订阅里程计，发布速度指令
    rospy.Subscriber("/odom", Odometry, odom_callback, queue_size=10)
    cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=50)
    rate = rospy.Rate(CONTROL_FREQUENCY)
    path_idx = 1  # 路径点索引，从第二个点开始跟踪

    rospy.loginfo("开始路径跟踪...")
    # 跟踪路径点
    while not rospy.is_shutdown() and path_idx < len(path_points):
        target_x, target_y = path_points[path_idx]
        # 计算当前点到目标点的距离和角度
        dx = target_x - current_x
        dy = target_y - current_y
        dist_to_target = math.hypot(dx, dy)

        # 到达当前路径点，切换下一个
        if dist_to_target < 0.1:
            path_idx += 1
            rospy.loginfo(f"到达路径点{path_idx-1}/{len(path_points)-1}")
            continue

        # 计算目标朝向，归一化到[-pi, pi]
        target_theta = math.atan2(dy, dx)
        angle_diff = target_theta - current_theta
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # 速度控制：比例系数调节，限制最大速度
        linear_vel = min(MAX_LINEAR, dist_to_target * 2)
        angular_vel = max(-MAX_ANGULAR, min(MAX_ANGULAR, angle_diff * 2.5))

        # 发布速度指令
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        cmd_pub.publish(cmd)
        rate.sleep()

    # 路径点跟踪完成，调整最终朝向至目标角度
    rospy.loginfo("路径点跟踪完成，调整最终朝向...")
    target_final_theta = math.radians(END_POINT[2])
    while not rospy.is_shutdown():
        angle_diff = target_final_theta - current_theta
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        # 朝向误差小于0.05rad（约2.86°），停止旋转
        if abs(angle_diff) < 0.05:
            break
        # 低速旋转，保证朝向精准
        angular_vel = max(-MAX_ANGULAR*0.4, min(MAX_ANGULAR*0.4, angle_diff * 2))
        cmd = Twist()
        cmd.angular.z = angular_vel
        cmd_pub.publish(cmd)
        rate.sleep()

    # 停止机器人
    cmd_pub.publish(Twist())
    rospy.loginfo("\n===== 导航完成！ =====")
    rospy.loginfo(f"机器人实际终点：({current_x:.3f}, {current_y:.3f})")
    rospy.loginfo(f"实际朝向：{math.degrees(current_theta):.1f}°")
    return True

# -------------------------- 主函数：流程入口 --------------------------
if __name__ == "__main__":
    try:
        # 初始化ROS节点
        rospy.init_node("turtlebot3_plaza_path_planning", anonymous=True)
        rospy.set_param('/use_sim_time', True)
        rospy.loginfo("等待Gazebo场景加载完成...")
        rospy.sleep(START_WAIT_TIME)

        # 1. 机器人复位到起点
        set_robot_initial_pose()
        rospy.sleep(1)

        # 2. 调用LLM生成路径（计时+Token统计）
        llm_start_time = time.time()
        raw_llm_path, llm_usage = get_llm_path() 
        llm_end_time = time.time()
        llm_cost = llm_end_time - llm_start_time

        if raw_llm_path and llm_usage:
            rospy.loginfo(f"glm-5.1路径规划成功，耗时：{llm_cost:.3f} 秒")
            rospy.loginfo("="*50)
            rospy.loginfo(f"本次规划总Token消耗：{llm_usage.total_tokens}")
            rospy.loginfo(f"  输入Token（场景描述+任务指令）：{llm_usage.prompt_tokens}")
            rospy.loginfo(f"  输出Token（路径+思考过程）：{llm_usage.completion_tokens}")
            rospy.loginfo("="*50)
        else:
            rospy.logerr(f"glm-5.1路径规划失败，耗时：{llm_cost:.3f} 秒")
            rospy.logerr("LLM生成路径失败，程序退出")
            exit(1)

        # 3. 解析并平滑路径
        raw_path_points = parse_path(raw_llm_path)
        if not raw_path_points:
            rospy.logerr("解析路径失败，程序退出")
            exit(1)
        smooth_path_points = interpolate_path(raw_path_points)

        # 4. 可视化路径和场景
        visualize_path(smooth_path_points)

        # 5. 验证路径有效性
        if not check_path_valid(smooth_path_points):
            rospy.logerr("路径存在碰撞，程序退出")
            exit(1)

        # 6. 路径跟踪运动控制
        path_to_gazebo_vel(smooth_path_points)

    except rospy.ROSInterruptException:
        rospy.logwarn("程序被ROS中断（如Ctrl+C）")
    except Exception as e:
        rospy.logerr(f"程序运行出错：{str(e)}", exc_info=True)
