import asyncio
import time

# Async function - can pause and resume
async def download(file_id: int) -> str:
    print(f"starting download {file_id}")
    await asyncio.sleep(1) #pause here, let other tasks run
    print(f"finished download {file_id}")
    return f"file_{file_id}"

# sequential async - one at a time
async def sequential() -> None:
    start: float = time.time()
    await download(1)
    await download(2)
    await download(3)
    end: float = time.time()
    print(f"sequential took {end - start: .2f} seconds\n")

#concurrent async - all at once
async def concurrent() -> None:
    start: float = time.time()
    results = await asyncio.gather(
        download(1),
        download(2),
        download(3)
    )
    end: float = time.time()
    print(f"results: {results}")
    print(f"concurrent took {end - start:.2f} seconds")

async def main() -> None:
    print("sequential: ")
    await sequential()
    print("concurrent: ")
    await concurrent()

asyncio.run(main())