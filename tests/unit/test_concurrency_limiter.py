"""Unit tests for concurrency_limiter module."""
import pytest
import asyncio
import threading
import time
from unittest.mock import MagicMock
from fastapi import HTTPException

from mlserver.concurrency_limiter import (
    PredictionSemaphore,
    PredictionLimiter,
    AsyncPredictionLimiter,
    create_prediction_limiter,
)


class TestPredictionSemaphore:
    """Test the PredictionSemaphore class."""

    def test_init_default(self):
        """Test default initialization."""
        sem = PredictionSemaphore()
        assert sem._max_concurrent == 1
        assert sem._timeout == 0
        assert sem.active_predictions == 0

    def test_init_custom(self):
        """Test custom initialization."""
        sem = PredictionSemaphore(max_concurrent=5, timeout=10)
        assert sem._max_concurrent == 5
        assert sem._timeout == 10

    def test_acquire_nowait_success(self):
        """Test successful immediate acquisition."""
        sem = PredictionSemaphore(max_concurrent=1)

        acquired = sem.acquire_nowait()
        assert acquired is True
        assert sem.active_predictions == 1

    def test_acquire_nowait_failure(self):
        """Test failed acquisition when full."""
        sem = PredictionSemaphore(max_concurrent=1)

        # First acquisition succeeds
        assert sem.acquire_nowait() is True

        # Second acquisition fails
        assert sem.acquire_nowait() is False
        assert sem.active_predictions == 1

    def test_release(self):
        """Test release functionality."""
        sem = PredictionSemaphore(max_concurrent=1)

        sem.acquire_nowait()
        assert sem.active_predictions == 1

        sem.release()
        assert sem.active_predictions == 0

    def test_multiple_concurrent(self):
        """Test with multiple concurrent slots."""
        sem = PredictionSemaphore(max_concurrent=3)

        assert sem.acquire_nowait() is True
        assert sem.acquire_nowait() is True
        assert sem.acquire_nowait() is True
        assert sem.active_predictions == 3

        # Fourth should fail
        assert sem.acquire_nowait() is False

        # Release one
        sem.release()
        assert sem.active_predictions == 2

        # Now can acquire again
        assert sem.acquire_nowait() is True

    def test_is_available(self):
        """Test is_available property."""
        sem = PredictionSemaphore(max_concurrent=1)

        assert sem.is_available is True
        sem.acquire_nowait()
        assert sem.is_available is False
        sem.release()
        assert sem.is_available is True

    def test_thread_safety(self):
        """Test thread safety of semaphore."""
        sem = PredictionSemaphore(max_concurrent=5)
        results = []

        def worker():
            if sem.acquire_nowait():
                time.sleep(0.01)
                results.append(True)
                sem.release()
            else:
                results.append(False)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should have been able to acquire at some point
        assert sem.active_predictions == 0


class TestPredictionLimiter:
    """Test the PredictionLimiter context manager."""

    def test_successful_enter_exit(self):
        """Test successful context manager usage."""
        sem = PredictionSemaphore(max_concurrent=1)

        with PredictionLimiter(sem) as limiter:
            assert sem.active_predictions == 1

        assert sem.active_predictions == 0

    def test_rejection_when_busy(self):
        """Test HTTP 503 raised when busy."""
        sem = PredictionSemaphore(max_concurrent=1)

        # First limiter acquires
        with PredictionLimiter(sem):
            # Second limiter should be rejected
            with pytest.raises(HTTPException) as exc_info:
                with PredictionLimiter(sem):
                    pass

            assert exc_info.value.status_code == 503
            assert "Retry-After" in exc_info.value.headers

    def test_custom_rejection_message(self):
        """Test custom rejection message."""
        sem = PredictionSemaphore(max_concurrent=1)
        sem.acquire_nowait()  # Block it

        custom_msg = "Custom busy message"
        with pytest.raises(HTTPException) as exc_info:
            with PredictionLimiter(sem, rejection_message=custom_msg):
                pass

        assert exc_info.value.detail == custom_msg

    def test_release_on_exception(self):
        """Test semaphore is released even on exception."""
        sem = PredictionSemaphore(max_concurrent=1)

        with pytest.raises(ValueError):
            with PredictionLimiter(sem):
                raise ValueError("Test error")

        assert sem.active_predictions == 0
        assert sem.is_available is True


class TestAsyncPredictionLimiter:
    """Test the AsyncPredictionLimiter async context manager."""

    @pytest.mark.asyncio
    async def test_successful_async_enter_exit(self):
        """Test successful async context manager usage."""
        sem = asyncio.Semaphore(1)

        async with AsyncPredictionLimiter(sem) as limiter:
            assert limiter._acquired is True

    @pytest.mark.asyncio
    async def test_async_rejection_when_busy(self):
        """Test HTTP 503 raised when async limiter is busy."""
        sem = asyncio.Semaphore(1)

        # Acquire the semaphore
        await sem.acquire()

        with pytest.raises(HTTPException) as exc_info:
            async with AsyncPredictionLimiter(sem):
                pass

        assert exc_info.value.status_code == 503

        # Release
        sem.release()

    @pytest.mark.asyncio
    async def test_async_release_on_exception(self):
        """Test semaphore is released even on async exception."""
        sem = asyncio.Semaphore(1)

        with pytest.raises(ValueError):
            async with AsyncPredictionLimiter(sem):
                raise ValueError("Async test error")

        # Semaphore should be available again
        # Try to acquire to verify
        acquired = await asyncio.wait_for(sem.acquire(), timeout=0.1)
        assert acquired is True
        sem.release()

    @pytest.mark.asyncio
    async def test_async_multiple_concurrent(self):
        """Test async limiter with multiple concurrent slots."""
        sem = asyncio.Semaphore(2)
        results = []

        async def worker(n):
            try:
                async with AsyncPredictionLimiter(sem, max_concurrent=2):
                    await asyncio.sleep(0.01)
                    results.append(f"success-{n}")
            except HTTPException:
                results.append(f"rejected-{n}")

        # Start 3 workers, 2 should succeed, 1 might be rejected
        await asyncio.gather(worker(1), worker(2), worker(3))

        # At least 2 should succeed
        successes = [r for r in results if r.startswith("success")]
        assert len(successes) >= 2


class TestCreatePredictionLimiter:
    """Test the create_prediction_limiter factory function."""

    def test_create_sync_limiter(self):
        """Test creating sync prediction limiter."""
        sem, limiter_factory = create_prediction_limiter(max_concurrent_predictions=2, async_mode=False)

        assert isinstance(sem, PredictionSemaphore)
        limiter = limiter_factory()
        assert isinstance(limiter, PredictionLimiter)

    def test_create_async_limiter(self):
        """Test creating async prediction limiter."""
        sem, limiter_factory = create_prediction_limiter(max_concurrent_predictions=3, async_mode=True)

        assert isinstance(sem, asyncio.Semaphore)
        limiter = limiter_factory()
        assert isinstance(limiter, AsyncPredictionLimiter)

    def test_sync_limiter_integration(self):
        """Test full sync limiter workflow."""
        sem, limiter_factory = create_prediction_limiter(max_concurrent_predictions=1, async_mode=False)

        with limiter_factory():
            assert sem.active_predictions == 1

        assert sem.active_predictions == 0

    @pytest.mark.asyncio
    async def test_async_limiter_integration(self):
        """Test full async limiter workflow."""
        sem, limiter_factory = create_prediction_limiter(max_concurrent_predictions=1, async_mode=True)

        async with limiter_factory():
            pass  # Just verify it works


class TestConcurrencyEdgeCases:
    """Test edge cases in concurrency limiting."""

    def test_zero_max_concurrent(self):
        """Test behavior with zero max concurrent (immediate blocking)."""
        # Note: This would always reject - testing for completeness
        sem = PredictionSemaphore(max_concurrent=0)
        assert sem.acquire_nowait() is False

    def test_high_concurrency(self):
        """Test with high concurrency limit."""
        sem = PredictionSemaphore(max_concurrent=100)

        for _ in range(100):
            assert sem.acquire_nowait() is True

        assert sem.active_predictions == 100
        assert sem.acquire_nowait() is False

        # Release all
        for _ in range(100):
            sem.release()

        assert sem.active_predictions == 0

    def test_limiter_reusability(self):
        """Test that limiters can be reused."""
        sem = PredictionSemaphore(max_concurrent=1)

        for i in range(5):
            with PredictionLimiter(sem):
                assert sem.active_predictions == 1
            assert sem.active_predictions == 0
