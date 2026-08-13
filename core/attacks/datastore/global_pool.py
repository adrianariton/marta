# import asyncio
# import time
# import concurrent.futures
# import threading
# from functools import wraps


# # --- GLOBAL BATCH MANAGER ---
# class GlobalBatchManager:
#     def __init__(self, max_batch_size=10, max_batch_time=5.0):
#         self.max_batch_size = max_batch_size
#         self.max_batch_time = max_batch_time
#         self.queue = asyncio.Queue()
#         # Thread pool to handle the blocking calls
#         self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)

#         # Start the consumer task
#         self.loop = asyncio.new_event_loop()
#         threading.Thread(target=self._run_event_loop, daemon=True).start()

#         # Start consumer within the loop
#         asyncio.run_coroutine_threadsafe(self._batch_consumer(), self.loop)

#     def _run_event_loop(self):
#         asyncio.set_event_loop(self.loop)
#         self.loop.run_forever()

#     async def _batch_consumer(self):
#         """Processes queue based on size or time limit."""
#         while True:
#             while self.queue.empty():
#                 await asyncio.sleep(0.1)

#             start_time = time.time()
#             while (
#                 self.queue.qsize() < self.max_batch_size
#                 and (time.time() - start_time) < self.max_batch_time
#             ):
#                 await asyncio.sleep(0.1)

#             print(f"--- Triggering Batch: Size={self.queue.qsize()} ---")

#             batch_items = []
#             while not self.queue.empty():
#                 batch_items.append(await self.queue.get())

#             futures_to_run = []
#             for func, args, kwargs, fut in batch_items:
#                 # Run sync function in thread pool
#                 task = self.loop.run_in_executor(self.executor, func, *args, **kwargs)
#                 futures_to_run.append((task, fut))

#             results = await asyncio.gather(*(t[0] for t in futures_to_run))

#             for (_, fut), res in zip(futures_to_run, results):
#                 if not fut.cancelled():
#                     self.loop.call_soon_threadsafe(fut.set_result, res)


# import os

# MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE")) or 3
# MAX_BATCH_TIME_S = float(os.getenv("MAX_BATCH_TIME_S")) or 5.0

# batch_manager = GlobalBatchManager(max_batch_size=MAX_BATCH_SIZE, max_batch_time=MAX_BATCH_TIME_S)


# def global_batched(func):
#     """Decorator that queues work but returns the result synchronously."""

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         self = args[0]
#         if hasattr(self, "use_batching") and self.use_batching:
#             fut = asyncio.run_coroutine_threadsafe(
#                 batch_manager.queue.put((func, args, kwargs, None)), batch_manager.loop
#             )

#             # --- CRITICAL CHANGE ---
#             # We create a new future in the calling context to wait on.
#             # This allows the manager to set the result later.
#             local_fut = asyncio.run_coroutine_threadsafe(
#                 batch_manager.loop.create_future(), batch_manager.loop
#             )
#             # Actually, to bridge synchronous and async properly here:
#             # We need a new type of future bridging.

#             # Let's use a standard threading event to block the thread
#             # and bridge the async result.

#             event = threading.Event()
#             result_container = []

#             async def put_and_wait():
#                 # Create a future for this specific call
#                 f = batch_manager.loop.create_future()
#                 await batch_manager.queue.put((func, args, kwargs, f))
#                 res = await f
#                 result_container.append(res)
#                 event.set()

#             asyncio.run_coroutine_threadsafe(put_and_wait(), batch_manager.loop)

#             # Block the synchronous thread until the async work is done
#             event.wait()
#             return result_container[0]

#         else:
#             return func(*args, **kwargs)

#     return wrapper


# # --- USAGE (Your existing loop remains unchanged) ---
# class TextGenerator:
#     def __init__(self, name, use_batching=True):
#         self.name = name
#         self.use_batching = use_batching

#     @global_batched
#     def generate(self, prompt):
#         print(f"[{self.name}] Running sync generation for '{prompt}'...")
#         time.sleep(1)  # Heavy blocking work
#         return f"Result for '{prompt}' from {self.name}"


# def main():
#     tgs = [TextGenerator(f"Model-{i}") for i in range(20)]
#     rs = []

#     start_time = time.time()
#     print("Starting loop...")
#     for i in range(20):
#         # Your original synchronous call
#         ri = tgs[i].generate(f"lalalala {i}")
#         rs.append(ri)
#         print(f"Got: {ri}")

#     end_time = time.time()
#     print(f"Total time: {end_time - start_time:.2f} seconds")


# if __name__ == "__main__":
#     main()
