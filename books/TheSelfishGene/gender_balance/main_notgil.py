import numpy as np
import matplotlib.pyplot as plt
import queue
import random
import threading
import time,sys

class Task:
    
    iterations = 400 # 迭代次数
    capacity = 1_000_000 # 生物的总容量
    threads = 5
    batch_size = int(capacity//(threads*16))# 每批次的任务数量
    mutation_rate = 0.05 # 变异率
    rng = np.random.default_rng()
    parent = rng.random(capacity,dtype=np.float32) # 父代生男性概率,前面是男性，后面是女性
    parent_male:int = 0 # 父代中男性结束的位置
    offspring = rng.random(capacity,dtype=np.float32) # 后代
    offspring_male:int = 0 # 后代中男性结束的位置
    offspring_female:int = capacity # 后代中女性开始的位置
    queue_male: queue.Queue[np.ndarray] = queue.Queue() # 男性队列
    queue_female: queue.Queue[np.ndarray] = queue.Queue() # 女性队列
    task_queue: queue.Queue['Task']= queue.Queue() # 任务队列

    returns_male:np.ndarray = np.zeros(iterations,dtype=np.float32) # 每次迭代的男性比例
    returns_variance:np.ndarray = np.zeros(iterations,dtype=np.float32)  # 方差


    def __init__(self,count:int):
        '''任务初始化        
        :param count: 任务数量
        '''
        self.count = count
        self._rng = np.random.default_rng() # 每个任务独立RNG，避免线程竞争


    def run(self):
        '''运行任务（向量化版本）'''
        count = self.count
        rng = self._rng
        # 批量随机选父亲和母亲
        p_males = self.parent[rng.integers(0, self.parent_male, size=count)]
        p_females = self.parent[rng.integers(self.parent_male, self.capacity, size=count)]
        # 批量计算后代基因值 + 变异
        p_offspring = (p_males + p_females) / 2 + rng.normal(0, self.mutation_rate, size=count).astype(np.float32)
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

def simulate(parent_male=0.5):
    Task.parent_male = int(Task.capacity * parent_male) # 设置初始父代男性数量
    Task.parent = Task.rng.normal(parent_male, 0.1, Task.capacity).astype(np.float32) # 重新生成父代概率
    # Task.offspring = Task.rng.random(Task.capacity,dtype=np.float32) # 重置后代数组

    for i in range(Task.iterations):
        count = Task.capacity
        while count > 0: # 分发任务
            batch = min(Task.batch_size, count)
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
        # 统计男性比例
        Task.returns_male[i] = Task.offspring_male / Task.capacity
        Task.returns_variance[i] = np.var(Task.offspring) *100 # 便展示
        if Task.returns_male[i] <=0.0 or Task.returns_male[i] >= 1.0:
            print(f"迭代{i}男性比例{Task.returns_male[i]:.4f}")
            break
        # 后代成为父代，进入下一轮迭代
        Task.parent, Task.offspring = Task.offspring, Task.parent
        Task.parent_male = Task.offspring_male
        Task.offspring_male = 0
        Task.offspring_female = Task.capacity
        if i % 20 == 0:
            print(f"迭代{i}完成，男性比例{Task.returns_male[i]:.4f}")



def show():
    print(f"最终男性比例{Task.returns_male[-1]:.4f}")
    plt.plot(Task.returns_male)
    plt.plot(Task.returns_variance)
    plt.xlabel("change iteration")
    plt.ylabel("male ratio")
    plt.title("male ratio change over iterations")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    t0=time.time()
    init_threads(Task.threads)
    parent_male = 0.9991 #Task.rng.random() # 初始父代男性比例
    print(f"初始父代男性比例{parent_male:.4f}")
    simulate(parent_male)
    print(f"总耗时{time.time() - t0:.2f}秒")
    show()


