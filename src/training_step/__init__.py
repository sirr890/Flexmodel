from training_step.training_module import Generic_trainings

# Define the public API of the package
__all__ = ["Generic_trainings"]

# Add initialization
PACKAGE_VERSION = "1.0.0"


def initialize():
    """Initialization logic for the package."""
    print(f"Initializing my_package version {PACKAGE_VERSION}")
