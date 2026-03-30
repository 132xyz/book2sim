import numpy as np
import matplotlib.pyplot as plt
import time

# 参数
ITERATIONS = 100
CAPACITY = 10_000
MUTATION_RATE = 0.01

rng = np.random.default_rng()

def simulate(parent_male_ratio=0.5):
    parent_male = int(CAPACITY * parent_male_ratio)
    parent = rng.normal(parent_male_ratio, 0.1, CAPACITY).astype(np.float32)
    offspring = np.zeros(CAPACITY, dtype=np.float32)
    returns_male = np.zeros(ITERATIONS, dtype=np.float32)

    for i in range(ITERATIONS):
        male_idx = 0
        female_idx = CAPACITY - 1

        for _ in range(CAPACITY):
            # 随机选一个父亲和一个母亲
            p_male = parent[rng.integers(0, parent_male)]
            p_female = parent[rng.integers(parent_male, CAPACITY)]
            # 平均 + 变异
            p_offspring = (p_male + p_female) / 2 + rng.normal(0, MUTATION_RATE)
            p_offspring = np.clip(p_offspring, 0.0, 1.0)
            # 判断子代性别
            if rng.random() < p_offspring:
                offspring[male_idx] = p_offspring
                male_idx += 1
            else:
                offspring[female_idx] = p_offspring
                female_idx -= 1

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