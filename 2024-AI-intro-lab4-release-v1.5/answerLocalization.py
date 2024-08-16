from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
k=1.2
pos_variance=0.1
theta_variance=0.15
def is_inside_walls(walls,p):
    # point_floor = np.floor(p)
    # point_remain = p - point_floor
    # occupied = []
    # if point_remain[0] < 0.75 and point_remain[1] < 0.75:
    #     occupied.append([point_floor[0], point_floor[1]])
    # if point_remain[0] < 0.75 and point_remain[1] > 0.25:
    #     occupied.append([point_floor[0], point_floor[1] + 1])
    # if point_remain[0] > 0.25 and point_remain[1] < 0.75:
    #     occupied.append([point_floor[0] + 1, point_floor[1]])
    # if point_remain[0] > 0.25 and point_remain[1] > 0.25:
    #     occupied.append([point_floor[0] + 1, point_floor[1] + 1])
    # for block in occupied:
    #     if block in walls:
    #         return True
    # return False
    for wall in walls:
        v=np.abs(p-wall)-0.5
        u=np.where(v>0,v,0)
        if u[0]*u[0]+u[1]*u[1]<0.0625:
            return True
        # if wall[0]-0.75<p[0]<wall[0]+0.75 and wall[1]-0.75<p[1]<wall[1]+0.75:
        #     return True
    return False
### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    max_xy=np.max(walls,axis=0)
    min_xy=np.min(walls,axis=0)
    all_particles: List[Particle] = []
    for _ in range(N):
        # all_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
        theta=np.random.uniform(-np.pi,np.pi)
        while True:
            pos=np.array([np.random.uniform(min_xy[0],max_xy[0]),np.random.uniform(min_xy[1],max_xy[1])])
            if not is_inside_walls(walls,pos):
                all_particles.append(Particle(pos[0], pos[1], theta, 1/N))
                break
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    weight=np.exp(-k*np.linalg.norm(estimated-gt))
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    # for _ in range(len(particles)):
    #     resampled_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    N=len(particles)
    r=np.random.uniform(0,1/N)
    c=particles[0].weight
    i=0
    # sum=0
    # for _ in range(N):
    #     sum+=particles[_].weight
    #     print(_,particles[_].weight)
    # print('sum=',sum)
    for _ in range(N):
        while r+_/N>c and i<N-1:
            i+=1
            # print('_=',_)
            # print(r+_/N,c)
            # print('N=',N)
            # print(i)
            c+=particles[i].weight
        while True:
            pos=particles[i].position+np.random.normal(0,pos_variance,2)
            if not is_inside_walls(walls,pos):
                theta=particles[i].theta+np.random.normal(0,theta_variance)
                # if theta>=np.pi:
                #     theta-=2*np.pi
                # elif theta<-np.pi:
                #     theta+=2*np.pi
                resampled_particles.append(Particle(pos[0], pos[1], theta, 1/N))
                break
    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.theta+=dtheta
    p.position+=traveled_distance*np.array([np.cos(p.theta),np.sin(p.theta)])
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###
    # n=len(particles)
    # for p in particles:
    #     final_result.position+=p.position/n
    #     final_result.theta+=p.weight*(p.theta-int(p.theta/(2*np.pi))*(2*np.pi))
    final_result.position=particles[0].position
    final_result.theta=particles[0].theta
    ### 你的代码 ###
    return final_result