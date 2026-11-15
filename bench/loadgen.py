"""
Async open-loop load generators for the backtest service.

Open loop: requests go out on a schedule fixed in advance, whether or not
earlier ones have answered, so a slow server can't slow the load down and
hide its own latency (coordinated omission). Latency is measured from the
scheduled send time as well as from the actual one; the two differ only
if the generator itself fell behind, and `max_send_lag_ms` reports by how
much.

    submit_poisson   POST /backtests, Poisson arrivals at a mean rate
    submit_burst     POST /backtests as fast as `concurrency` allows
    get_fixed_rate   GET /backtests/{id} at a fixed rate
"""

import asyncio
import random
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx2

# The fixed strategy mix: every strategy with its default parameters, in
# turn, over the same SPY window. Each request gets its own seed, so every
# job is a distinct request (no idempotent replays) doing the same work.
STRATEGIES = ("sma_crossover", "momentum", "mean_reversion", "ml_direction")
TICKER, START, END = "SPY", "2010-01-01", "2025-01-01"


def request_body(i: int) -> dict[str, Any]:
    return {"strategy": STRATEGIES[i % len(STRATEGIES)], "ticker": TICKER,
            "start": START, "end": END, "seed": i}


@dataclass
class Sample:
    scheduled: float          # monotonic time the request was due
    sent: float
    done: float
    status: int | None        # None: transport error
    job_id: str | None = None

    @property
    def latency_ms(self) -> float:
        return (self.done - self.scheduled) * 1000

    @property
    def service_ms(self) -> float:
        return (self.done - self.sent) * 1000


@dataclass
class Run:
    samples: list[Sample] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0

    @property
    def max_send_lag_ms(self) -> float:
        return max(((s.sent - s.scheduled) * 1000 for s in self.samples), default=0.0)

    @property
    def errors(self) -> int:
        return sum(1 for s in self.samples if s.status is None or s.status >= 400)


def _client(base_url: str, connections: int) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        base_url=base_url, timeout=httpx2.Timeout(30.0),
        limits=httpx2.Limits(max_connections=connections,
                             max_keepalive_connections=connections))


async def _post(client: httpx2.AsyncClient, i: int, scheduled: float) -> Sample:
    sent = time.monotonic()
    try:
        response = await client.post("/backtests", json=request_body(i),
                                     headers={"Idempotency-Key": f'"{uuid.uuid4()}"'})
    except httpx2.TransportError:
        return Sample(scheduled, sent, time.monotonic(), None)
    job_id = response.json().get("id") if response.status_code == 202 else None
    return Sample(scheduled, sent, time.monotonic(), response.status_code, job_id)


async def _open_loop(schedule: Sequence[float], make: Callable[[int, float], Any]) -> Run:
    """Starts make(i, due) at each offset in `schedule` (seconds from now)."""
    run = Run(started=time.monotonic())
    tasks = []
    for i, offset in enumerate(schedule):
        due = run.started + offset
        delay = due - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(make(i, due)))
    run.samples = list(await asyncio.gather(*tasks))
    run.finished = time.monotonic()
    return run


def poisson_schedule(rate: float, duration: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    times, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= duration:
            return times
        times.append(t)


async def submit_poisson(base_url: str, rate: float, duration: float, seed: int,
                         first_index: int = 0, connections: int = 64) -> Run:
    async with _client(base_url, connections) as client:
        return await _open_loop(poisson_schedule(rate, duration, seed),
                                lambda i, due: _post(client, first_index + i, due))


async def submit_burst(base_url: str, n: int, first_index: int = 0,
                       concurrency: int = 32) -> Run:
    """n POSTs, `concurrency` at a time: the backlog for a throughput run."""
    run = Run(started=time.monotonic())
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(n):
        queue.put_nowait(first_index + i)

    async with _client(base_url, concurrency) as client:
        async def sender() -> None:
            while not queue.empty():
                i = queue.get_nowait()
                run.samples.append(await _post(client, i, time.monotonic()))

        await asyncio.gather(*(sender() for _ in range(concurrency)))
    run.finished = time.monotonic()
    return run


async def get_fixed_rate(base_url: str, job_ids: Sequence[str], rate: float, duration: float,
                         seed: int, connections: int = 64) -> Run:
    rng = random.Random(seed)
    ids = [rng.choice(job_ids) for _ in range(int(rate * duration))]

    async with _client(base_url, connections) as client:
        async def get(i: int, due: float) -> Sample:
            sent = time.monotonic()
            try:
                response = await client.get(f"/backtests/{ids[i]}")
            except httpx2.TransportError:
                return Sample(due, sent, time.monotonic(), None)
            return Sample(due, sent, time.monotonic(), response.status_code, ids[i])

        return await _open_loop([i / rate for i in range(len(ids))], get)
