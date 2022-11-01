"adapted from https://github.com/ClashLuke/SharedUtils"
import multiprocessing
import numpy as np
import time
import typing
import uuid
from multiprocessing.shared_memory import SharedMemory


def return_false():
    return False


def call_with(
    contexts: list,
    fn: typing.Callable[[], None],
    cond_fn: typing.Callable[[], bool] = return_false,
    retry: typing.Union[bool, int] = False,
):
    """
    Used to call a function with multiple context objects (for example, 4 multiprocessing locks) while ensuring a
    condition stays false before entering the next context.
    This is useful for when slowly acquiring multiple locks in a highly distributed setting. Sometimes it can be the
    case that the reason those locks should've been acquired was already fixed by another process,
    so that slowly acquiring the rest of the locks isn't necessary anymore.
    :param contexts: A list of objects that can be entered using a with statement (needs __enter__ and __exit__,
    such as tf.control_dependencies and multiprocessing.Lock())
    :param fn: Function that will be called once all contexts are entered
    :param cond_fn: Callback called whenever a context is acquired, to ensure the function still has to run.
    :param retry: whether to try acquiring all locks again, or to raise an error (or number of retries, -2 = infinity)
    :return: either none (-> cond = True) or output of fn
    """
    if retry is False:
        retry = 0
    elif retry is True:
        retry = -2
    while retry != -1:
        retry -= 1
        if not contexts:
            return fn()
        if cond_fn():
            return None

        try:
            with contexts[0]:
                return call_with(contexts[1:], fn, cond_fn, False)
        except Exception as exc:  # pylint: disable=broad-except
            if retry == -1:
                raise exc


class Timeout(multiprocessing.TimeoutError):
    pass


class ListQueue:
    """
    A reimplementation of multiprocessing's Queue with a public list attribute which can be inspected by any process
    without forcibly locking the whole thing.
    It's used the same way as a normal queue, with the slight difference that ListQueue has a `.list` attribute which
    displays tbe entire queue in order.
    """

    def __init__(self, timeout: float = 0):
        manager = multiprocessing.Manager()
        self.list = manager.list()  # type: ignore
        self.lock_writing = manager.Value(bool, False)
        self.write_lock = manager.RLock()
        self.read_lock = manager.RLock()
        self.cond = manager.Condition(manager.Lock())
        self.timeout = timeout

    def get(self):
        with self.read_lock:
            if not self.list:
                with self.cond:
                    if not self._cond.wait(self.timeout if self.timeout > 0 else None):
                        raise Timeout
            return self.list.pop(0)

    def put(self, obj):
        with self.write_lock:
            self.list.append(obj)
            with self.cond:
                self.cond.notify_all()


class FiFoSemaphoreTimeout(multiprocessing.TimeoutError):
    pass


class FiFoSemaphore:
    """
    A semaphore that processes items in a FiFo fashion with optional values to increment or decrement by.
    This can be useful to process things in order and have cleaner retry-loops.
    Additionally, it's used by `SharedSequentialQueue` to "enqueue" that one worker can't have any other worker to run.
    Once all other workers are finished, this main worker would be started immediately and does its task. Without
    values, this would require many calls to .acquire() where any of the other workers could intervene and cause a
    deadlock. Without the in-order execution, it's possible that tasks don't get executed in the FiFo way they were
    intended by the programmer.
    Usage:
    >>> def worker(semaphore: FiFoSemaphore):
    ...     with semaphore(5):
    ...         print("Hello World")
    ...         semaphore.acquire(10)
    ...         print("Fully acquired")
    ...         semaphore.release(5)
    ...         print("Releasing 5")
    ...         semaphore.release(5)
    ...         print("Releasing 5 more")
    ...     print("Exit")
    >>> sem = FiFoSemaphore(15)
    >>> [multiprocessing.Process(target=worker, args=(sem,)).start() for _ in range(3)]
    Hello World
    Fully acquired
    Releasing 5
    Hello World
    Releasing 5 more
    Hello World
    Exit
    Fully acquired
    Releasing 5
    Fully acquired
    Releasing 5 more
    Releasing 5
    Exit
    Exit
    """

    def __init__(self, value: int = 1, timeout: float = 0):
        manager = multiprocessing.Manager()
        self._cond = multiprocessing.Condition(multiprocessing.Lock())
        self._value = manager.list([value])
        self._queue = ListQueue()
        self._value_lock = multiprocessing.Lock()
        self.max_value = value
        self.timeout = timeout

    def __call__(self, val: int = 0):
        return FiFoSemaphoreContext(self, val)

    def acquire(self, val: int = 0):
        "acquire lock"
        job_id = uuid.uuid4()
        self._queue.put(job_id)
        if val < 1:
            val = self.max_value + val
        with self._cond:
            while self._queue.list[0] != job_id or self._value[0] < val:
                if not self._cond.wait(self.timeout if self.timeout > 0 else None):
                    raise Timeout
            with self._value_lock:
                self._value[0] -= val
            self._queue.get()
        return True

    def release(self, val: int = 0):
        "release lock"
        if val < 1:
            val = self.max_value + val
        with self._cond:
            with self._value_lock:
                self._value[0] += val
            self._cond.notify_all()


class FiFoSemaphoreContext:
    def __init__(self, semaphore: FiFoSemaphore, val: int):
        self.semaphore = semaphore
        self.val = val

    def __enter__(self):
        self.semaphore.acquire(self.val)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release(self.val)


class SharedQueue:
    """
    The memory allocation is similar to the ExtQueue above, however, this queue simulates single-threaded access by
    fully locking access to the internal memory when one thread is reading or writing to it.
    It might still give speedups compared to multiprocessing queues, as the copy overhead is lower. However, it won't be
    as fast as the ExtQueue, which locks only a sequence of memory at the cost of decreased safety.
    """

    data_mem: SharedMemory
    data: np.ndarray
    lock: FiFoSemaphore
    retry: bool
    index_queue: ListQueue

    @classmethod
    def from_shape(cls, *shape: int, dtype: np.dtype = np.dtype(np.uint8), timeout: float = 1.0, retry: bool = True):
        self = cls()
        self.data_mem = SharedMemory(create=True, size=np.zeros(shape, dtype=dtype).nbytes)
        self.data = np.ndarray(shape, dtype=dtype, buffer=self.data_mem.buf)
        self.lock = FiFoSemaphore(1, timeout)
        self.retry = retry
        self.index_queue = ListQueue(timeout)
        return self

    @classmethod
    def from_export(cls, data_name, shape, dtype, lock, retry, index_queue):
        self = cls()
        self.data_mem = SharedMemory(create=False, name=data_name)
        self.data = np.ndarray(shape, dtype=dtype, buffer=self.data_mem.buf)
        self.lock = lock
        self.retry = retry
        self.index_queue = index_queue
        return self

    def export(self):
        return self.data_mem.name, self.data.shape, self.data.dtype, self.lock, self.retry, self.index_queue

    def _get_data(self):
        start, end, info = self.index_queue.get()
        return self.data[start:end].copy(), info  # local clone, so share can be safely edited

    def get(self):
        ret = call_with([self.lock()], self._get_data, retry=self.retry)
        if self.index_queue.lock_writing.value and not self:
            self.index_queue.lock_writing.value = False
            time.sleep(1)
        return ret

    def _shift_left(self):
        local_list = list(self.index_queue.list)
        min_start = local_list[0][0]
        max_end = local_list[-1][1]

        if max_end - min_start > 0.75 * self.data.shape[0]:
            self.index_queue.lock_writing.value = True
        self.data[: max_end - min_start] = self.data[min_start:max_end]
        self.index_queue.list[:] = [(start - min_start, end - min_start, info) for start, end, info in local_list]

    def _put_item(self, obj: np.ndarray, info: dict):
        batches = obj.shape[0]
        if self.index_queue.list:
            _, start, _ = self.index_queue.list[-1]
            start += 1
        else:
            start = 0
        end = start + batches
        self.index_queue.put((start, end, info))
        self.data[start:end] = obj[:]

    def _fits(self, batches: int):
        return not self or self.index_queue.list[-1][1] + batches < self.data.shape[0]

    def _write(self, obj: np.ndarray, info: dict):
        if not self._fits(obj.shape[0]):
            self._shift_left()
        if not self._fits(obj.shape[0]):
            raise ValueError("Doesn't fit after shift.")
        self._put_item(obj, info)

    def put(self, obj: np.ndarray, info: dict):
        while self.index_queue.lock_writing.value:
            time.sleep(5)
        call_with([self.lock()], lambda: self._write(obj, info), retry=self.retry)

    def __bool__(self):
        return bool(self.index_queue.list)
