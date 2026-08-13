from typing import TYPE_CHECKING
import asyncio
import time
from typing import List, TYPE_CHECKING

# This block is ignored at runtime
if TYPE_CHECKING:
    from core.attacks.interfaces import TextGenerator, OneShotConversation


# class Batch:
#     def __init__(
#         self, max_batch_size: int = 5, max_wait_seconds: float = 5.0, check_frequency: float = 0.1
#     ):
#         self.generator: "TextGenerator" = None
#         self.task_queue: List[asyncio.Future] = []
#         self.conversations: List["OneShotConversation"] = []

#         self.max_batch_size = max_batch_size
#         self.max_wait_seconds = max_wait_seconds
#         self.check_frequency = check_frequency

#         self._last_flush_time = time.time()
#         self._lock = asyncio.Lock()
#         self._flush_task = None

#     def attach_generator(self, generator: "TextGenerator"):
#         self.generator = generator

#     def enqueue_and_wait(self, task: "OneShotConversation"):
#         raise Exception(
#             "Sync batching is not supported, please try async (await textgenerator.agenerate, or await textgenerator.aapply_attack)"
#         )

#     async def aenqueue_and_wait(self, task: "OneShotConversation"):
#         if self._flush_task is None or self._flush_task.done():
#             self._flush_task = asyncio.create_task(self._timer_loop())

#         loop = asyncio.get_running_loop()
#         future = loop.create_future()

#         async with self._lock:
#             self.task_queue.append(future)
#             self.conversations.append(task)
#             if len(self.conversations) >= self.max_batch_size:
#                 await self._flush()

#         return await future

#     async def _timer_loop(self):
#         """Periodically checks if we need to flush based on time."""
#         while True:
#             await asyncio.sleep(self.check_frequency)  # Check every 100ms
#             async with self._lock:
#                 if not self.conversations:
#                     continue
#                 if (time.time() - self._last_flush_time) > self.max_wait_seconds:
#                     await self._flush()

#     async def _flush(self):
#         """The actual logic that calls the generator."""
#         if not self.conversations:
#             return
#         current_convs = self.conversations
#         current_futures = self.task_queue
#         self.conversations = []
#         self.task_queue = []
#         self._last_flush_time = time.time()
#         try:
#             results = await self.generator.abatch_generate(current_convs)
#             print(f"\t>Generated results for {len(current_convs)} tasks.", flush=True)
#             for future, result in zip(current_futures, results):
#                 if not future.done():
#                     future.set_result(result)
#         except Exception as e:
#             for future in current_futures:
#                 if not future.done():
#                     future.set_exception(e)

#     async def end(self):
#         """
#         Manually flushes the remaining queue and stops the background timer.
#         Use this when shutting down your server or finishing a script.
#         """
#         async with self._lock:
#             if self.conversations:
#                 print(f"Final flush: processing {len(self.conversations)} remaining tasks.")
#                 await self._flush()

#             # Stop the background timer task gracefully
#             if self._flush_task and not self._flush_task.done():
#                 self._flush_task.cancel()
#                 try:
#                     await self._flush_task
#                 except asyncio.CancelledError:
#                     pass
#                 self._flush_task = None


import asyncio
import time
from typing import List, TYPE_CHECKING, Optional

# This block is ignored at runtime
if TYPE_CHECKING:
    from core.attacks.interfaces import TextGenerator, OneShotConversation


class Batch:
    def __init__(
        self,
        max_batch_size: int = 30,
        max_wait_seconds: float = 18.0,
        check_frequency: float = 0.5,  # Slightly less aggressive check for vLLM stability
        name: str = "DefaultBatch",
    ):
        self.generator: Optional["TextGenerator"] = None
        self.task_queue: List[asyncio.Future] = []
        self.conversations: List["OneShotConversation"] = []

        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds
        self.check_frequency = check_frequency

        # Use monotonic for time-delta calculations (immune to system clock resets)
        self._last_flush_time = time.monotonic()
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

        self.name = name

    def attach_generator(self, generator: "TextGenerator"):
        self.generator = generator

    def enqueue_and_wait(self, task: "OneShotConversation"):
        raise Exception(
            "Sync batching is not supported, please try async (await textgenerator.agenerate)"
        )

    async def aenqueue_and_wait(self, task: "OneShotConversation"):
        # Start timer heartbeat if not running
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._timer_loop())

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        should_flush_immediately = False

        async with self._lock:
            # RESET TIMER: If this is the first task in a new batch,
            # start the wait countdown from NOW.
            if not self.conversations:
                self._last_flush_time = time.monotonic()

            self.task_queue.append(future)
            self.conversations.append(task)

            if len(self.conversations) >= self.max_batch_size:
                should_flush_immediately = True

        # Perform the flush outside the lock to avoid deadlocks
        # and allow next-batch tasks to queue up while vLLM is busy.
        if should_flush_immediately:
            await self._flush()

        return await future

    async def _timer_loop(self):
        """Periodically checks if the current batch has timed out."""
        try:
            while True:
                await asyncio.sleep(self.check_frequency)

                async with self._lock:
                    if not self.conversations:
                        continue

                    # Check if the oldest item in the current batch has waited too long
                    if (time.monotonic() - self._last_flush_time) < self.max_wait_seconds:
                        continue

                # If we've reached here, timeout has expired.
                # Call flush outside the lock block.
                await self._flush()
        except asyncio.CancelledError:
            pass

    async def _flush(self):
        """Logic to send the batch to the generator."""
        current_convs = []
        current_futures = []

        # 1. Atomic State Swap: Grab the current tasks and clear the shared state.
        async with self._lock:
            if not self.conversations:
                return

            current_convs = self.conversations
            current_futures = self.task_queue

            # Reset queues for the next batch immediately
            self.conversations = []
            self.task_queue = []
            # Reset flush time for the next potential batch
            self._last_flush_time = time.monotonic()

        # 2. Network/API Call: Performed without holding the lock.
        # This allows your 30 parallel loops to start queuing their NEXT turn.
        try:
            if self.generator is None:
                raise ValueError("Generator not attached to Batch.")

            results = await self.generator.abatch_generate(current_convs)

            print(
                f"\t [Batch {self.name}] Generated results for {len(current_convs)} tasks.",
                flush=True,
            )

            for future, result in zip(current_futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            for future in current_futures:
                if not future.done():
                    future.set_exception(e)

    async def end(self):
        """Shutdown: Process remaining tasks and stop the timer."""
        # Process what's left
        await self._flush()

        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
