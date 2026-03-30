import numpy as np
import matplotlib.pyplot as plt
import time

# 参数
ITERATIONS = 1000
CAPACITY = 1_000_000
MUTATION_RATE = 0.05

rng = np.random.default_rng()

def simulate(parent_male_ratio=0.5):
    parent_male = int(CAPACITY * parent_male_ratio)
    parent = rng.normal(parent_male_ratio, 0.1, CAPACITY).astype(np.float32)
    offspring = np.zeros(CAPACITY, dtype=np.float32)
    returns_male = np.zeros(ITERATIONS, dtype=np.float32)

    for i in range(ITERATIONS):
        # 批量随机选父亲和母亲
        p_males = parent[rng.integers(0, parent_male, size=CAPACITY)]
        p_females = parent[rng.integers(parent_male, CAPACITY, size=CAPACITY)]
        # 批量计算后代基因值 + 变异
        p_all = (p_males + p_females) / 2 + rng.normal(0, MUTATION_RATE, size=CAPACITY).astype(np.float32)
        np.clip(p_all, 0.0, 1.0, out=p_all)
        # 批量判断性别
        is_male = rng.random(CAPACITY, dtype=np.float32) < p_all
        males = p_all[is_male]
        females = p_all[~is_male]
        male_idx = len(males)
        # 写入offspring数组（男性在前，女性在后）
        offspring[:male_idx] = males
        offspring[CAPACITY - len(females):] = females

        # 统计男性比例
        returns_male[i] = male_idx / CAPACITY
        if returns_male[i] <= 0.0 or returns_male[i] >= 1.0:
            print(f"迭代{i} 男性比例{returns_male[i]:.4f}")
            break
        # 后代变父代
        parent, offspring = offspring, parent
        parent_male = male_idx

    return returns_male

def show(returns_male):
    print(f"最终男性比例{returns_male[-1]:.4f}")
    plt.plot(returns_male)
    plt.xlabel("iteration")
    plt.ylabel("male ratio")
    plt.title("male ratio change over iterations")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    t0 = time.time()
    returns_male = simulate(parent_male_ratio=0.1)
    print(f"总耗时{time.time() - t0:.2f}秒")
    show(returns_male)