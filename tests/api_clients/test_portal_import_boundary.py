import builtins
import importlib
import sys


def test_portal_clients_do_not_import_optional_integrations(monkeypatch):
    blocked_modules = (
        "susse.api_clients.merra_2",
        "susse.api_clients.modis",
        "susse.estimators",
    )
    original_import = builtins.__import__

    for module_name in list(sys.modules):
        if module_name == "susse" or module_name.startswith("susse."):
            sys.modules.pop(module_name)

    def guarded_import(name, *args, **kwargs):
        if name.startswith(blocked_modules):
            raise AssertionError(f"Portal client import loaded optional module {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    api_clients = importlib.import_module("susse.api_clients")

    assert api_clients.CAMSClient
    assert api_clients.NASAPowerFetchData
    assert api_clients.NASAPowerProduct
    assert api_clients.TemporalResolution
