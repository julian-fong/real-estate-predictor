import asyncio
import queue
import argparse
from model import main

parser  = argparse.ArgumentParser(description = 'Enter the amount of days to retrieve information from')
parser.add_argument("days", help='A natural number for the number of days to retrieve information from', metavar='days')
parser.add_argument("-l", "--lease", action="store_true", help='specify if you want to generate a new model specifically for lease data')
parser.add_argument("-s", "--sale", action="store_true", help='specify if you want to generate a new model specifically for sale data')
parser.add_argument("--m", "--model", action = "store_true", help='specify if you want to generate a model')

# parser.add_argument('--type', metavar='type', type = str, nargs = 1, help = 'A type (optional) specifying to grab lease or sale data', required = False,)

#parser.add_argument("days")

args = parser.parse_args()
print(args)

target_dir = args.days
print(target_dir)

print(args.lease)
print(args.sale)
print(args.model)

# target_dir1 = Path(args.sale)

# async def task1(q):
#     print("Task 1 started")
#     my_list = [1, 2, 3, 4, 5]
#     for i in my_list:
#         await asyncio.sleep(1)
#         q.put(i)
#     print("Task 1 completed")

# async def task2(q):
#     print("Task 2 started")
#     while True:
#         try:
#             value = q.get_nowait()
#             print("Task 2 received value:", value)
#         except queue.Empty:
#             print("Task 2 waiting for value...")
#             await asyncio.sleep(1)
#             continue
#         if value is None:
#             print("Task 2 completed")
#             break

# async def main():
#     q = asyncio.Queue()
#     # Run both tasks concurrently
#     task1_coro = asyncio.create_task(task1(q))
#     task2_coro = asyncio.create_task(task2(q))

#     # Wait for task1 to complete
#     await task1_coro

#     # Signal task2 to complete
#     await q.put(None)
#     await task2_coro

# # # Start the event loop and run the main coroutine
# # asyncio.run(main())


# if __name__ == "__main__":