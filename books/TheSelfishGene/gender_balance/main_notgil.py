import numpy as np
import matplotlib.pyplot as plt
import queue
import threading
import time,sys


class Task:
    
    iterations = 500 # 迭代次数
    max_capacity = 1_000_000 # 最大容量（不变）
    capacity = max_capacity//2 # 当前种群大小（可变，允许萎缩）
    threads = 5
    mutation_rate = 0.05 # 变异率
    male_rate:float = 100.0 # 每只雄性平均繁殖次数
    female_rate:float = 1.71 # 每只雌性平均繁殖次数
    gen_male_rate = 0.49
    rng = np.random.default_rng()
    parent = rng.random(max_capacity,dtype=np.float32) # 父代生男性概率,前面是男性，后面是女性
    parent_male:int = 0 # 父代中男性结束的位置
    offspring = rng.random(max_capacity,dtype=np.float32) # 后代
    offspring_male:int = 0 # 后代中男性结束的位置
    offspring_female:int = max_capacity # 后代中女性开始的位置
    queue_male: queue.Queue[np.ndarray] = queue.Queue() # 男性队列
    queue_female: queue.Queue[np.ndarray] = queue.Queue() # 女性队列
    task_queue: queue.Queue['Task']= queue.Queue() # 任务队列

    returns_male:np.ndarray = np.zeros(iterations,dtype=np.float32) # 每次迭代的男性比例
    returns_variance:np.ndarray = np.zeros(iterations,dtype=np.float32)  # 方差
    returns_population:np.ndarray = np.zeros(iterations,dtype=np.int32) # 每次迭代的种群数量


    def __init__(self,count:int):
        '''任务初始化
        
        :param count: int: 本批次后代数量
        '''
        self.count = count
        self._rng = np.random.default_rng() # 每个任务独立RNG，避免线程竞争


    def run(self):
        '''运行任务（向量化版本）'''
        count = self.count
        rng = self._rng
        # 批量随机选父亲和母亲（均匀采样，平均繁殖率由总量控制）
        p_males = self.parent[rng.integers(0, self.parent_male, size=count)]
        p_females = self.parent[rng.integers(self.parent_male, self.capacity, size=count)]
        # 批量计算后代基因值 + 变异
        p_offspring = p_males*Task.gen_male_rate+ p_females*(1-Task.gen_male_rate) + rng.normal(0, self.mutation_rate, size=count).astype(np.float32)
        np.clip(p_offspring, 0.0, 1.0, out=p_offspring)
        # 批量判断性别
        is_male = rng.random(count, dtype=np.float32) < p_offspring
        # 任务完成放入队列
        self.queue_male.put(p_offspring[is_male])
        self.queue_female.put(p_offspring[~is_male])

def run_thread():
    '''运行线程，处理任务队列'''
    task_queue = Task.task_queue
    while True:
        task = task_queue.get()
        if task is None:
            break
        task.run()

def init_threads(num_threads:int):
    '''初始化线程池
    
    :param num_threads: 线程数量
    '''
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=run_thread,daemon=True)
        t.start()
        threads.append(t)
    return threads

def simulate(parent_male:float=0.5, initial_population:int=0):
    '''运行模拟
    
    :param parent_male: float: 初始父代男性比例
    :param initial_population: int: 初始种群大小，0表示使用最大容量
    '''
    Task.capacity = int(initial_population) if initial_population > 0 else Task.max_capacity
    Task.parent_male = int(Task.capacity * parent_male) # 设置初始父代男性数量
    Task.parent[:Task.capacity] = Task.rng.normal(parent_male, 0.1, Task.capacity).astype(np.float32) # 重新生成父代概率

    for i in range(Task.iterations):
        n_males = Task.parent_male
        n_females = Task.capacity - Task.parent_male

        if n_males == 0 or n_females == 0:
            print(f"迭代{i} 种群灭绝: 雄性{n_males}, 雌性{n_females}")
            break

        # 计算本代实际后代数（受平均繁殖率约束）
        actual = int(min(n_males * Task.male_rate, n_females * Task.female_rate, Task.max_capacity))
        if actual <= 0:
            print(f"迭代{i} 种群无法繁殖: 雄性{n_males}, 雌性{n_females}")
            break
        batch_size = max(1, actual // (Task.threads * 16))

        # 分发任务（每个线程独立均匀采样父母，平均繁殖率由总量自然控制）
        Task.offspring_male = 0
        Task.offspring_female = actual
        count = actual
        while count > 0:
            batch = min(batch_size, count)
            Task.task_queue.put(Task(batch))
            count -= batch
        
        # 收集结果
        while 0 < Task.offspring_female - Task.offspring_male:
            empty = False
            try:
                male = Task.queue_male.get(timeout=1)
                size = len(male)
                Task.offspring[Task.offspring_male:Task.offspring_male + size] = male
                Task.offspring_male += size
            except queue.Empty:
                empty = True

            try:
                female = Task.queue_female.get(timeout=1)
                size = len(female)
                Task.offspring[Task.offspring_female - size:Task.offspring_female] = female
                Task.offspring_female -= size
            except queue.Empty:
                empty = True

            if empty:
                time.sleep(0.001) # 短暂等待任务完成
                print(f"迭代{i}等待任务完成，剩余:{Task.offspring_female - Task.offspring_male}")

        if Task.offspring_female - Task.offspring_male != 0:
            raise ValueError(f"后代数量不匹配{Task.offspring_female} , {Task.offspring_male}")

        # 统计
        Task.returns_male[i] = Task.offspring_male / actual
        Task.returns_variance[i] = np.var(Task.offspring[:actual]) * 100
        Task.returns_population[i] = actual

        if Task.returns_male[i] <=0.0 or Task.returns_male[i] >= 1.0:
            print(f"迭代{i}男性比例{Task.returns_male[i]:.4f}, 种群:{actual}")
            break

        # 后代成为父代，更新容量
        Task.parent, Task.offspring = Task.offspring, Task.parent
        Task.parent_male = Task.offspring_male
        Task.capacity = actual
        Task.offspring_male = 0
        Task.offspring_female = Task.capacity

        if i % 20 == 0:
            print(f"迭代{i}完成，男性比例{Task.returns_male[i]:.4f}, 种群:{actual}")



def show():
    valid = Task.returns_population > 0
    print(f"最终男性比例{Task.returns_male[valid][-1]:.4f}, 种群:{Task.returns_population[valid][-1]}")

    fig, ax1 = plt.subplots()
    ax1.plot(Task.returns_male[valid], 'b-', label='male ratio')
    ax1.plot(Task.returns_variance[valid], 'g-', alpha=0.5, label='variance x100')
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("male ratio / variance", color='b')
    ax1.axhline(y=0.5, color='b', linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(np.where(valid)[0], Task.returns_population[valid], 'r-', alpha=0.5, label='population')
    ax2.set_ylabel("population", color='r')

    plt.title("gender balance simulation (multi-thread, birth limit)")
    ax1.grid()
    plt.show()

if __name__ == "__main__":
    t0=time.time()
    init_threads(Task.threads)
    parent_male = 0.55 #Task.rng.random() # 初始父代男性比例
    initial_pop = int(Task.max_capacity/1.1) # 初始种群大小，0=最大容量
    print(f"初始父代男性比例{parent_male:.4f}, 初始种群:{initial_pop or Task.max_capacity}, 雄性繁殖率:{Task.male_rate}, 雌性繁殖率:{Task.female_rate}")
    simulate(parent_male, initial_pop)
    print(f"总耗时{time.time() - t0:.2f}秒")
    show()


