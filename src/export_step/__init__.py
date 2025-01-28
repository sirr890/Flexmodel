from export_step.export_module import ExportationGeneric

# Define the public API of the package
__all__ = ["ExportationGeneric"]

# Add initialization
PACKAGE_VERSION = "1.0.0"


def initialize():
    """Initialization logic for the package."""
    print(f"Initializing my_package version {PACKAGE_VERSION}")
