"""Top-level SuSSE exports loaded only when requested."""

from importlib import import_module

_EXPORTS = {
    "DEFAULT_IDENTIFIER": ".api_clients",
    "EMAIL_ENV_KEY": ".api_clients",
    "ENV_PATH": ".api_clients",
    "TIMEOUT_SECONDS": ".api_clients",
    "CAMSClient": ".api_clients",
    "Merra2Config": ".api_clients",
    "MerraDataFetcher": ".api_clients",
    "MerraDownloadManager": ".api_clients",
    "MerraProducts": ".api_clients",
    "MerraStreamConfig": ".api_clients",
    "MerraDataStreamFetcher": ".api_clients",
    "StreamSessionManager": ".api_clients",
    "ModisConfig": ".api_clients",
    "ModisDataFetcher": ".api_clients",
    "ModisBand": ".api_clients",
    "ModisProdFrequency": ".api_clients",
    "ModisProduct": ".api_clients",
    "ModisProductEnum": ".api_clients",
    "ModisProductFactory": ".api_clients",
    "NASAPowerConfig": ".api_clients",
    "NASAPowerFetchData": ".api_clients",
    "NASAPowerResult": ".api_clients",
    "NASAPowerProduct": ".api_clients",
    "TemporalResolution": ".api_clients",
    "ClearSkyEstimate": ".estimators.clearsky",
    "ClearSkyEstimator": ".estimators.clearsky",
    "ClearSkyEstimatorPVlib": ".estimators.clearsky",
    "REST2Model": ".estimators.clearsky",
    "StatisticalMetrics": ".estimators.validation",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
