import numpy as np
import matplotlib.pyplot as plt
import time

# 参数
ITERATIONS = 400
MAX_CAPACITY = 1_000_000
MUTATION_RATE = 0.05
MALE_RATE = 100.0   # 每只雄性平均繁殖次数
FEMALE_RATE = 1.5   # 每只雌性平均繁殖次数

rng = np.random.default_rng()


def simulate(parent_male_ratio:float=0.5, initial_population:int=0):
    """运行模拟

    :param parent_male_ratio: float: 初始父代男性比例
    :param initial_population: int: 初始种群大小，0表示使用最大容量
    :return: tuple: (男性比例数组, 种群数量数组)
    """
    capacity = int(initial_population) if initial_population > 0 else MAX_CAPACITY
    parent_male = int(capacity * parent_male_ratio)
    parent = rng.normal(parent_male_ratio, 0.1, capacity).astype(np.float32)
    offspring = np.zeros(MAX_CAPACITY, dtype=np.float32)
    returns_male = np.zeros(ITERATIONS, dtype=np.float32)
    returns_population = np.zeros(ITERATIONS, dtype=np.int32)

    for i in range(ITERATIONS):
        n_males = parent_male
        n_females = capacity - parent_male

        if n_males == 0 or n_females == 0:
            print(f"迭代{i} 种群灭绝: 雄性{n_males}, 雌性{n_females}")
            break

        # 计算本代实际后代数（受平均繁殖率约束）
        actual = int(min(n_males * MALE_RATE, n_females * FEMALE_RATE, MAX_CAPACITY))
        if actual <= 0:
            print(f"迭代{i} 种群无法繁殖: 雄性{n_males}, 雌性{n_females}")
            break

        # 批量随机选父亲和母亲（均匀采样，平均繁殖率由总量控制）
        p_males = parent[rng.integers(0, n_males, size=actual)]
        p_females = parent[rng.integers(n_males, capacity, size=actual)]

        # 批量计算后代基因值 + 变异
        p_all = (p_males + p_females) / 2 + rng.normal(0, MUTATION_RATE, size=actual).astype(np.float32)
        np.clip(p_all, 0.0, 1.0, out=p_all)

        # 批量判断性别
        is_male = rng.random(actual, dtype=np.float32) < p_all
        males = p_all[is_male]
        females = p_all[~is_male]
        male_count = len(males)

        # 写入offspring数组（男性在前，女性在后）
        offspring[:male_count] = males
        offspring[actual - len(females):actual] = females

        # 统计
        returns_male[i] = male_count / actual
        returns_population[i] = actual

        if returns_male[i] <= 0.0 or returns_male[i] >= 1.0:
            print(f"迭代{i} 男性比例{returns_male[i]:.4f}, 种群:{actual}")
            break

        # 后代变父代，更新容量
        parent, offspring = offspring, parent
        parent_male = male_count
        capacity = actual

        if i % 20 == 0:
            print(f"迭代{i} 男性比例{returns_male[i]:.4f}, 种群:{actual}")

    return returns_male, returns_population


def show(returns_male:np.ndarray, returns_population:np.ndarray):
    """显示结果图表

    :param returns_male: np.ndarray: 每代男性比例
    :param returns_population: np.ndarray: 每代种群数量
    """
    valid = returns_population > 0
    print(f"最终男性比例{returns_male[valid][-1]:.4f}, 种群:{returns_population[valid][-1]}")

    fig, ax1 = plt.subplots()
    ax1.plot(returns_male[valid], 'b-', label='male ratio')
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("male ratio", color='b')
    ax1.axhline(y=0.5, color='b', linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(np.where(valid)[0], returns_population[valid], 'r-', alpha=0.5, label='population')
    ax2.set_ylabel("population", color='r')

    plt.title("gender balance simulation (single thread)")
    ax1.grid()
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    initial_pop = 0 # 初始种群大小，0=最大容量
    print(f"初始种群:{initial_pop or MAX_CAPACITY}, 雄性繁殖率:{MALE_RATE}, 雌性繁殖率:{FEMALE_RATE}")
    returns_male, returns_pop = simulate(parent_male_ratio=0.991, initial_population=initial_pop)
    print(f"总耗时{time.time() - t0:.2f}秒")
    show(returns_male, returns_pop)