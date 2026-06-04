"""Public API client exports loaded only when requested.

Keeping these imports lazy lets downstream applications install only the client
dependencies they use without importing unrelated optional integrations.
"""

from importlib import import_module

_EXPORTS = {
    "DEFAULT_IDENTIFIER": ".cams",
    "EMAIL_ENV_KEY": ".cams",
    "ENV_PATH": ".cams",
    "TIMEOUT_SECONDS": ".cams",
    "CAMSClient": ".cams",
    "Merra2Config": ".merra_2",
    "MerraDataFetcher": ".merra_2",
    "MerraDownloadManager": ".merra_2",
    "MerraProducts": ".merra_2",
    "MerraStreamConfig": ".merra_2",
    "MerraDataStreamFetcher": ".merra_2",
    "StreamSessionManager": ".merra_2",
    "ModisConfig": ".modis",
    "ModisDataFetcher": ".modis",
    "ModisBand": ".modis",
    "ModisProdFrequency": ".modis",
    "ModisProduct": ".modis",
    "ModisProductEnum": ".modis",
    "ModisProductFactory": ".modis",
    "NASAPowerConfig": ".NASA_Power",
    "NASAPowerFetchData": ".NASA_Power",
    "NASAPowerResult": ".NASA_Power",
    "NASAPowerProduct": ".NASA_Power",
    "TemporalResolution": ".NASA_Power",
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
