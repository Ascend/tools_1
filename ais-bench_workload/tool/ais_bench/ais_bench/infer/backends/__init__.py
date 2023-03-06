import os

from ais_bench.infer import registry

BACKEND_REGISTRY = registry.Registry("BACKEND_REGISTRY")

registry.import_all_modules_for_register(
    os.path.dirname(os.path.abspath(__file__)), "ais_bench.infer.backends"
)


class BackendFactory:
    @staticmethod
    def create_backend(name):
        return BACKEND_REGISTRY[name]
