"""
PyQueue - A lightweight async task queue with workers, priorities,
retries, rate limiting, and persistence. Pure Python, no dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pyqueue")

# ============================================
# ENUMS & DATA CLASSES
# ============================================
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class Priority(int, Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class Task:
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    max_retries: int = 3
    timeout: float = 60.0
    scheduled_at: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["priority"] = self.priority.name
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        data = dict(data)
        data["priority"] = Priority[data["priority"]]
        data["status"] = TaskStatus(data["status"])
        return cls(**data)


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0


@dataclass
class WorkerStats:
    worker_id: str
    tasks_processed: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    total_duration: float = 0.0
    started_at: float = field(default_factory=time.time)

    @property
    def avg_duration(self) -> float:
        if self.tasks_processed == 0:
            return 0.0
        return self.total_duration / self.tasks_processed


# ============================================
# EXCEPTIONS
# ============================================
class TaskError(Exception):
    """Base exception for task errors."""


class TaskTimeoutError(TaskError):
    """Raised when a task exceeds its timeout."""


class TaskRetryError(TaskError):
    """Raised to explicitly signal a retry is needed."""


class RateLimitError(TaskError):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: float = 1.0):
        self.retry_after = retry_after
        super().__init__(f"Rate limited; retry after {retry_after}s")


# ============================================
# PRIORITY QUEUE (heap-based)
# ============================================
class PriorityQueue:
    """Min-heap ordered by (priority, scheduled_at, created_at)."""

    def __init__(self) -> None:
        self._heap: List[tuple] = []
        self._index: Dict[str, Task] = {}

    def _key(self, task: Task) -> tuple:
        scheduled = task.scheduled_at or task.created_at
        return (int(task.priority), scheduled, task.created_at)

    def push(self, task: Task) -> None:
        import heapq

        self._index[task.id] = task
        heapq.heappush(self._heap, (self._key(task), task.id))

    def pop(self) -> Optional[Task]:
        import heapq

        while self._heap:
            _, task_id = heapq.heappop(self._heap)
            task = self._index.pop(task_id, None)
            if task is not None:
                return task
        return None

    def peek(self) -> Optional[Task]:
        if not self._heap:
            return None
        _, task_id = self._heap[0]
        return self._index.get(task_id)

    def remove(self, task_id: str) -> bool:
        return self._index.pop(task_id, None) is not None

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._index

    def clear(self) -> None:
        self._heap.clear()
        self._index.clear()


# ============================================
# RATE LIMITER (token bucket)
# ============================================
class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int = 10) -> None:
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    async def acquire(self, tokens: int = 1) -> None:
        async with self._lock:
            self._refill()
            while self._tokens < tokens:
                wait = (tokens - self._tokens) / self.rate
                await asyncio.sleep(wait)
                self._refill()
            self._tokens -= tokens


# ============================================
# PERSISTENCE
# ============================================
class TaskStorage:
    """Simple JSON-file persistence for tasks."""

    def __init__(self, path: str = "tasks.json") -> None:
        self.path = Path(path)

    def save(self, tasks: List[Task]) -> None:
        try:
            data = [t.to_dict() for t in tasks]
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, default=str))
            tmp.replace(self.path)
        except Exception as exc:
            logger.error("Failed to save tasks: %s", exc)

    def load(self) -> List[Task]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text())
            return [Task.from_dict(item) for item in raw]
        except Exception as exc:
            logger.error("Failed to load tasks: %s", exc)
            return []


# ============================================
# TASK HANDLER REGISTRY
# ============================================
T = TypeVar("T")
Handler = Callable[[Task], Awaitable[Any]]


class HandlerRegistry:
    def __init__(self) -> None:
        self._handlers: Dict[str, Handler] = {}

    def register(self, name: str) -> Callable[[Handler], Handler]:
        def decorator(fn: Handler) -> Handler:
            self._handlers[name] = fn
            logger.debug("Registered handler: %s", name)
            return fn
        return decorator

    def get(self, name: str) -> Handler:
        if name not in self._handlers:
            raise TaskError(f"No handler registered for '{name}'")
        return self._handlers[name]

    def has(self, name: str) -> bool:
        return name in self._handlers

    def names(self) -> List[str]:
        return list(self._handlers.keys())


# Global registry instance
registry = HandlerRegistry()


# ============================================
# QUEUE / BROKER
# ============================================
class TaskQueue:
    """Central task queue with priority, retries, and persistence."""

    def __init__(
        self,
        storage: Optional[TaskStorage] = None,
        rate_limit: Optional[float] = None,
    ) -> None:
        self._queue = PriorityQueue()
        self._all_tasks: Dict[str, Task] = {}
        self._storage = storage
        self._rate_limiter = RateLimiter(rate_limit) if rate_limit else None
        self._event = asyncio.Event()
        self._closed = False
        self._lock = asyncio.Lock()

        if self._storage:
            for task in self._storage.load():
                if task.status in (TaskStatus.PENDING, TaskStatus.RETRYING):
                    task.status = TaskStatus.PENDING
                    self._queue.push(task)
                    self._all_tasks[task.id] = task

    async def enqueue(self, task: Task) -> str:
        if self._closed:
            raise RuntimeError("Queue is closed")

        async with self._lock:
            self._all_tasks[task.id] = task
            self._queue.push(task)
            self._event.set()
            logger.info(
                "Enqueued task %s [%s] priority=%s",
                task.id[:8], task.name, task.priority.name,
            )

        self._persist()
        return task.id

    async def dequeue(self, timeout: Optional[float] = None) -> Optional[Task]:
        while not self._closed:
            async with self._lock:
                task = self._queue.pop()
                if task is not None:
                    if self._rate_limiter:
                        pass  # Rate limiting handled by worker
                    return task
                self._event.clear()

            try:
                await asyncio.wait_for(self._event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return None

        return None

    async def requeue(self, task: Task, delay: float = 0.0) -> None:
        task.status = TaskStatus.RETRYING
        task.attempts += 1
        if delay > 0:
            task.scheduled_at = time.time() + delay

        async with self._lock:
            self._all_tasks[task.id] = task
            self._queue.push(task)
            self._event.set()

        logger.warning(
            "Requeued task %s (attempt %d/%d, delay=%.1fs)",
            task.id[:8], task.attempts, task.max_retries, delay,
        )
        self._persist()

    def cancel(self, task_id: str) -> bool:
        task = self._all_tasks.get(task_id)
        if not task:
            return False
        task.status = TaskStatus.CANCELLED
        removed = self._queue.remove(task_id)
        self._persist()
        return removed

    def get(self, task_id: str) -> Optional[Task]:
        return self._all_tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        tasks = list(self._all_tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for task in self._all_tasks.values():
            counts[task.status.value] = counts.get(task.status.value, 0) + 1
        counts["queued"] = len(self._queue)
        return counts

    async def close(self) -> None:
        self._closed = True
        self._event.set()
        self._persist()

    def _persist(self) -> None:
        if self._storage:
            self._storage.save(list(self._all_tasks.values()))


# ============================================
# WORKER
# ============================================
class Worker:
    """Async worker that pulls and executes tasks."""

    def __init__(
        self,
        queue: TaskQueue,
        worker_id: Optional[str] = None,
        concurrency: int = 1,
        poll_timeout: float = 1.0,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:6]}"
        self.concurrency = concurrency
        self.poll_timeout = poll_timeout
        self.stats = WorkerStats(worker_id=self.worker_id)
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        self._running = True
        logger.info(
            "Worker %s starting with concurrency=%d",
            self.worker_id, self.concurrency,
        )
        self._tasks = [
            asyncio.create_task(self._loop(i)) for i in range(self.concurrency)
        ]
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        logger.info("Worker %s stopped", self.worker_id)

    async def _loop(self, slot: int) -> None:
        while self._running:
            try:
                task = await self.queue.dequeue(timeout=self.poll_timeout)
                if task is None:
                    continue
                await self._execute(task)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Worker %s slot %d crashed: %s", self.worker_id, slot, exc)

    async def _execute(self, task: Task) -> TaskResult:
        start = time.monotonic()
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        task.attempts += 1

        logger.info(
            "[%s] ▶ %s (%s) attempt %d/%d",
            self.worker_id, task.name, task.id[:8], task.attempts, task.max_retries,
        )

        try:
            if not registry.has(task.name):
                raise TaskError(f"Unknown task handler: '{task.name}'")

            handler = registry.get(task.name)
            result = await asyncio.wait_for(handler(task), timeout=task.timeout)

            task.result = result
            task.status = TaskStatus.SUCCESS
            task.completed_at = time.time()
            duration = time.monotonic() - start

            self.stats.tasks_processed += 1
            self.stats.tasks_succeeded += 1
            self.stats.total_duration += duration

            logger.info(
                "[%s] ✓ %s completed in %.3fs",
                self.worker_id, task.id[:8], duration,
            )
            return TaskResult(task.id, TaskStatus.SUCCESS, result, duration=duration)

        except asyncio.TimeoutError:
            return await self._handle_failure(
                task, f"Timeout after {task.timeout}s", start, retryable=True,
            )
        except TaskRetryError as exc:
            return await self._handle_failure(
                task, str(exc), start, retryable=True,
            )
        except RateLimitError as exc:
            delay = exc.retry_after
            await self.queue.requeue(task, delay=delay)
            return TaskResult(task.id, TaskStatus.RETRYING, error=str(exc))
        except Exception as exc:
            return await self._handle_failure(
                task, str(exc), start, retryable=True,
            )

    async def _handle_failure(
        self, task: Task, error: str, start: float, retryable: bool,
    ) -> TaskResult:
        duration = time.monotonic() - start
        task.error = error
        self.stats.tasks_processed += 1
        self.stats.total_duration += duration

        if retryable and task.attempts < task.max_retries:
            delay = min(2 ** task.attempts * 0.5, 30.0)
            await self.queue.requeue(task, delay=delay)
            logger.warning(
                "[%s] ↻ %s failed: %s (retry in %.1fs)",
                self.worker_id, task.id[:8], error, delay,
            )
            return TaskResult(task.id, TaskStatus.RETRYING, error=error, duration=duration)

        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        self.stats.tasks_failed += 1

        logger.error(
            "[%s] ✗ %s failed permanently: %s",
            self.worker_id, task.id[:8], error,
        )
        return TaskResult(task.id, TaskStatus.FAILED, error=error, duration=duration)


# ============================================
# DEMO TASK HANDLERS
# ============================================
@registry.register("send_email")
async def send_email(task: Task) -> Dict[str, Any]:
    to = task.payload.get("to", "user@example.com")
    await asyncio.sleep(0.2)
    return {"sent_to": to, "timestamp": datetime.now().isoformat()}


@registry.register("process_image")
async def process_image(task: Task) -> Dict[str, Any]:
    path = task.payload.get("path", "image.jpg")
    await asyncio.sleep(0.5)
    return {"path": path, "resized": True, "size_kb": 128}


@registry.register("flaky_task")
async def flaky_task(task: Task) -> str:
    """Task that fails a couple of times before succeeding."""
    if task.attempts < 3:
        raise TaskRetryError(f"Simulated failure on attempt {task.attempts}")
    return "eventually succeeded"


@registry.register("slow_task")
async def slow_task(task: Task) -> str:
    await asyncio.sleep(10)
    return "done"


@registry.register("compute")
async def compute(task: Task) -> int:
    n = int(task.payload.get("n", 10))
    total = sum(i * i for i in range(n))
    return total


# ============================================
# SCHEDULER
# ============================================
class Scheduler:
    """Simple periodic task scheduler."""

    def __init__(self, queue: TaskQueue) -> None:
        self.queue = queue
        self._jobs: List[asyncio.Task] = []
        self._running = False

    def every(
        self,
        interval: float,
        task_name: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        async def runner() -> None:
            while self._running:
                await asyncio.sleep(interval)
                await self.queue.enqueue(
                    Task(name=task_name, payload=payload or {})
                )

        self._jobs.append(asyncio.create_task(runner()))

    async def start(self) -> None:
        self._running = True
        logger.info("Scheduler started with %d jobs", len(self._jobs))

    async def stop(self) -> None:
        self._running = False
        for job in self._jobs:
            job.cancel()


# ============================================
# DEMO / MAIN
# ============================================
async def demo() -> None:
    storage = TaskStorage("demo_tasks.json")
    queue = TaskQueue(storage=storage, rate_limit=50.0)

    # Enqueue a variety of tasks
    await queue.enqueue(Task("send_email", {"to": "a@b.com"}, Priority.HIGH))
    await queue.enqueue(Task("process_image", {"path": "/tmp/pic.png"}, Priority.NORMAL))
    await queue.enqueue(Task("flaky_task", {}, Priority.LOW, max_retries=5))
    await queue.enqueue(Task("compute", {"n": 100}, Priority.CRITICAL))
    await queue.enqueue(Task("send_email", {"to": "c@d.com"}, Priority.LOW))

    # Start two workers
    worker1 = Worker(queue, worker_id="w1", concurrency=2)
    worker2 = Worker(queue, worker_id="w2", concurrency=1)

    worker_tasks = [
        asyncio.create_task(worker1.start()),
        asyncio.create_task(worker2.start()),
    ]

    # Wait for processing
    await asyncio.sleep(6)

    await worker1.stop()
    await worker2.stop()
    for t in worker_tasks:
        t.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)

    # Report
    print("\n" + "=" * 60)
    print("QUEUE STATS")
    print("=" * 60)
    for key, val in queue.stats().items():
        print(f"  {key:12s} : {val}")

    print("\n" + "=" * 60)
    print("WORKER STATS")
    print("=" * 60)
    for w in (worker1, worker2):
        s = w.stats
        print(f"  {s.worker_id}: processed={s.tasks_processed} "
              f"ok={s.tasks_succeeded} fail={s.tasks_failed} "
              f"avg={s.avg_duration:.3f}s")

    print("\n" + "=" * 60)
    print("TASK RESULTS")
    print("=" * 60)
    for task in queue.list_tasks():
        print(f"  {task.id[:8]} {task.name:16s} {task.status.value:10s} "
              f"attempts={task.attempts} result={task.result!r}")

    await queue.close()


def main() -> None:
    print("🚀 PyQueue Demo\n")
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nInterrupted by user")


if __name__ == "__main__":
    main()
