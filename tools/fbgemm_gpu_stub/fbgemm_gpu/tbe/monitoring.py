"""Stub for fbgemm_gpu.tbe.monitoring."""


class TBEStatsReporterConfig:
    """Config for TBE stats reporting."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class AsyncSeriesTimer:
    """Stub timer for async series profiling."""

    def __init__(self, *args, **kwargs):
        pass
