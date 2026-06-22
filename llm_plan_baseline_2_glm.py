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

# -------------------------- 核心配置 --------------------------
DASHSCOPE_API_KEY = "sk-fba66d331d824c36ae5ff30960c93aea"
dashscope.api_key = DASHSCOPE_API_KEY


START_POINT = (-1.0,-1.0,90)# (1.5,0,90) (-1.0,-1.0,90)
END_POINT =(0,0,0)#(0,1.5,0) (0,0,0)


# 正方形围栏 4 面墙（真实结构：中心x, 中心y, 长度, 厚度, 旋转角度）
WALLS = [
    {"x": 0,     "y": 1.925,  "length": 4, "thick": 0.15, "angle": 0},    
    {"x": -1.925,"y": 0,      "length": 4, "thick": 0.15, "angle": 90},    
    {"x": 0,     "y": -1.925, "length": 4, "thick": 0.15, "angle": 0},   
    {"x": 1.925, "y": 0,      "length": 4, "thick": 0.15, "angle": 90}    
]

# 新增：Gazebo中的4个圆柱障碍物
CYLINDERS = [
    {"x": -0.6, "y": -0.6, "radius": 0.15},
    {"x": -0.6, "y":  0.6, "radius": 0.15},
    {"x":  0.6, "y": -0.6, "radius": 0.15},
    {"x":  0.6, "y":  0.6, "radius": 0.15},
]

SAFE_DISTANCE = 0.15
ROBOT_RADIUS = 0.10  # TurtleBot3半径

# 运动控制参数
MAX_LINEAR = 0.22
MAX_ANGULAR = 1.82
CONTROL_FREQUENCY = 20
START_WAIT_TIME = 3

current_x = START_POINT[0]
current_y = START_POINT[1]
current_theta = math.radians(START_POINT[2])

# -------------------------- 可视化：画墙 + 圆柱 --------------------------
def visualize_path(path_points):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # 画 4 面真实矩形墙
    for wall in WALLS:
        x = wall["x"]
        y = wall["y"]
        L = wall["length"]
        T = wall["thick"]
        angle = wall["angle"]

        if angle == 0:
            rx = x - L/2
            ry = y - T/2
            w = L
            h = T
        else:
            rx = x - T/2
            ry = y - L/2
            w = T
            h = L

        rect = patches.Rectangle((rx, ry), w, h, color='red', alpha=0.5)
        ax.add_patch(rect)

    # 画4个圆柱障碍物
    for cyl in CYLINDERS:
        circle = patches.Circle((cyl["x"], cyl["y"]), cyl["radius"], 
                               color='blue', alpha=0.6)
        ax.add_patch(circle)

    # 画路径
    if path_points:
        x_coords = [p[0] for p in path_points]
        y_coords = [p[1] for p in path_points]
        path_line, = ax.plot(x_coords, y_coords, 'g-', linewidth=2, label='Planned Path')
        ax.scatter(x_coords, y_coords, c='green', s=30)

    # 起点终点
    ax.scatter(START_POINT[0], START_POINT[1], c='lime', s=120, 
               label=f'Start ({START_POINT[0]}, {START_POINT[1]})')
    ax.scatter(END_POINT[0], END_POINT[1], c='orange', s=120, 
               label=f'End ({END_POINT[0]}, {END_POINT[1]})')

    ax.legend(loc='upper right')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('TurtleBot3 + Square Wall + 4 Cylinders')
    ax.grid(True)
    plt.show(block=True)

# -------------------------- 碰撞检测：墙 + 圆柱 --------------------------
def is_point_collide_wall(px, py, wall):
    wx = wall["x"]
    wy = wall["y"]
    L = wall["length"]
    T = wall["thick"]
    angle = wall["angle"]

    dx = abs(px - wx)
    dy = abs(py - wy)

    if angle == 0:
        return dx <= L/2 + 0.01 and dy <= T/2 + 0.01
    else:
        return dx <= T/2 + 0.01 and dy <= L/2 + 0.01

def is_point_collide_cylinder(px, py, cyl):
    # 计算点到圆心距离
    dx = px - cyl["x"]
    dy = py - cyl["y"]
    dist = math.hypot(dx, dy)
    # 安全距离 = 圆柱半径 + 机器人半径 + 缓冲
    safe = cyl["radius"] + ROBOT_RADIUS + 0.05
    return dist < safe

def check_path_valid(path_points):
    if not path_points:
        return False

    start_ok = math.isclose(path_points[0][0], START_POINT[0], abs_tol=0.2) and \
               math.isclose(path_points[0][1], START_POINT[1], abs_tol=0.2)
    end_ok = math.isclose(path_points[-1][0], END_POINT[0], abs_tol=0.2) and \
             math.isclose(path_points[-1][1], END_POINT[1], abs_tol=0.2)

    if not start_ok or not end_ok:
        rospy.logerr("起点或终点不匹配！")
        return False

    # 检查撞墙
    for (px, py) in path_points:
        for wall in WALLS:
            if is_point_collide_wall(px, py, wall):
                rospy.logerr(f"碰撞墙壁！点 ({px:.2f}, {py:.2f})")
                return False

    # 检查撞圆柱
    for (px, py) in path_points:
        for cyl in CYLINDERS:
            if is_point_collide_cylinder(px, py, cyl):
                rospy.logerr(f"碰撞圆柱障碍！点 ({px:.2f}, {py:.2f}) 靠近圆柱 ({cyl['x']}, {cyl['y']})")
                return False

    rospy.loginfo("路径验证通过：无碰撞围墙 + 无碰撞圆柱！")
    return True

# -------------------------- 机器人复位 --------------------------
def set_robot_initial_pose():
    rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
    try:
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        initial_state = ModelState()
        initial_state.model_name = 'turtlebot3_burger'

        initial_state.pose.position.x = START_POINT[0]
        initial_state.pose.position.y = START_POINT[1]
        initial_state.pose.position.z = 0.0

        yaw = math.radians(START_POINT[2])
        initial_state.pose.orientation.z = math.sin(yaw / 2)
        initial_state.pose.orientation.w = math.cos(yaw / 2)
        initial_state.reference_frame = 'world'

        resp = set_model_state(initial_state)
        if resp.success:
            rospy.loginfo(f"机器人复位成功！")
            rospy.loginfo(f"起点：({START_POINT[0]}, {START_POINT[1]})，朝向：{START_POINT[2]}°")
    except Exception as e:
        rospy.logerr(f"复位失败：{e}")

# -------------------------- LLM路径规划（已更新提示词） --------------------------
def get_llm_path():
    prompt = """
你是TurtleBot3 Burger运动规划师，仅返回数字组合，无任何其他文字/符号/说明！
场景：正方形围栏 + 4个圆柱障碍物
围墙坐标：
Wall_0: (0, 1.925)
Wall_1: (-1.925, 0)
Wall_2: (0, -1.925)
Wall_3: (1.925, 0)
墙体尺寸：长4m、厚0.15m

4个圆柱障碍物（中心坐标，半径0.15m）：
Cyl1: (-0.6, -0.6)
Cyl2: (-0.6, 0.6)
Cyl3: (0.6, -0.6)
Cyl4: (0.6, 0.6)

所有障碍静态不可移动。
任务：从A(-1.0,-1.0,90)到B(0,0,0) ，绝对不碰撞任何围墙和圆柱！
要求：
1. 必须严格避开4个圆柱和4面围墙
2. 路径点步长最大0.1米，平滑
3. 输出格式：(x1,y1) → (x2,y2) → ... → (xn,yn)
4. 优先到达终点，严格避障
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
            extra_body={"enable_thinking": True, "reasoning_effort": "high"}
        )
        raw = response.choices[0].message.content.strip()
        rospy.loginfo(f"glm-5.1 返回路径：{raw}")
        return raw
    except Exception as e:
        rospy.logerr(f"glm-5.1 调用错误：{e}")
        return None

# -------------------------- 路径解析 --------------------------
def parse_path(raw_path):
    if not raw_path:
        return None
    # 兼容两种格式：(x,y) 和 x,y
    pattern = re.compile(r'\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?')
    points = pattern.findall(raw_path)
    if not points:
        rospy.logerr("未解析到路径点，原始返回内容：" + raw_path)
        return None
    return [(float(x), float(y)) for x, y in points]

# -------------------------- 路径平滑 --------------------------
def interpolate_path(path_points, step=0.1):
    if len(path_points) < 2:
        return path_points
    smooth = [path_points[0]]
    for i in range(1, len(path_points)):
        x0, y0 = path_points[i-1]
        x1, y1 = path_points[i]
        d = math.hypot(x1-x0, y1-y0)
        n = max(1, int(d/step))
        for j in range(1, n+1):
            f = j/n
            smooth.append((x0 + f*(x1-x0), y0 + f*(y1-y0)))
    return smooth

# -------------------------- 里程计 --------------------------
def odom_callback(msg):
    global current_x, current_y, current_theta
    current_x = msg.pose.pose.position.x
    current_y = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    current_theta = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

# -------------------------- 闭环控制 --------------------------
def path_to_gazebo_vel(path_points):
    global current_x, current_y, current_theta
    rospy.Subscriber("/odom", Odometry, odom_callback)
    cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=50)
    rate = rospy.Rate(CONTROL_FREQUENCY)
    idx = 1

    while not rospy.is_shutdown() and idx < len(path_points):
        tx, ty = path_points[idx]
        dx = tx - current_x
        dy = ty - current_y
        dist = math.hypot(dx, dy)
        if dist < 0.1:
            idx += 1
            continue

        target_angle = math.atan2(dy, dx)
        da = target_angle - current_theta
        da = (da + math.pi) % (2*math.pi) - math.pi

        v = min(MAX_LINEAR, dist * 2)
        w = max(-MAX_ANGULAR, min(MAX_ANGULAR, da * 2.5))
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        cmd_pub.publish(cmd)
        rate.sleep()

    # 最终朝向
    target_final = math.radians(END_POINT[2])
    while not rospy.is_shutdown():
        da = target_final - current_theta
        da = (da + math.pi) % (2*math.pi) - math.pi
        if abs(da) < 0.05:
            break
        cmd = Twist()
        cmd.angular.z = max(-MAX_ANGULAR*0.4, min(MAX_ANGULAR*0.4, da*2))
        cmd_pub.publish(cmd)
        rate.sleep()

    cmd_pub.publish(Twist())
    rospy.loginfo("\n导航完成！")
    rospy.loginfo(f"机器人实际终点：({current_x:.3f}, {current_y:.3f})")
    rospy.loginfo(f"实际朝向：{math.degrees(current_theta):.1f}°")
    return True

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    try:
        rospy.init_node("turtlebot3_square_wall", anonymous=True)
        rospy.set_param('/use_sim_time', True)
        rospy.sleep(START_WAIT_TIME)

        set_robot_initial_pose()
        rospy.sleep(1)

        # LLM路径规划计时
        llm_start_time = time.time()
        raw = get_llm_path()
        llm_end_time = time.time()
        llm_cost = llm_end_time - llm_start_time
        rospy.loginfo(f"LLM路径规划耗时：{llm_cost:.3f} 秒")

        if not raw:
            exit(1)

        path = parse_path(raw)
        if not path:
            exit(1)

        path = interpolate_path(path)
        visualize_path(path)

        if not check_path_valid(path):
            rospy.logerr("路径撞墙或圆柱！")
            exit(1)

        path_to_gazebo_vel(path)

    except rospy.ROSInterruptException:
        rospy.logwarn("程序中断")
    except Exception as e:
        rospy.logerr(f"错误：{e}")
