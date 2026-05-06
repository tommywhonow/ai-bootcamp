import multiprocessing
import time

#CPU bound task
def heavy_computation(n: int) -> int:
    total: int =0
    for i in range(n):
        total += i
    return total
if __name__ == "__main__":
    # Sequential
    print("sequential: ")
    start: float = time.time()
    for _ in range(4):
        heavy_computation(5_000_000)
    end: float = time.time()
    print(f"took {end-start: .2f} seconds\n")

    #multiprocessing
    print("multiprocessing: ")
    start = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(heavy_computation, [5_000_000] * 4)
    end = time.time()
    print(f"took {end - start: .2f} seconds")
