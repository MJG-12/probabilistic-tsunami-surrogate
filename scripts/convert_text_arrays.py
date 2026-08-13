from probabilistic_tsunami_surrogate.config import RunConfig
from probabilistic_tsunami_surrogate.data.preprocessing import convert_txt_to_npy


def convert_text_arrays(config):
    """Converts simulation text arrays to faster NumPy arrays."""
    converted = convert_txt_to_npy(config.DATA_ROOT)
    print(f"Converted {len(converted)} arrays")


if __name__ == "__main__":
    convert_text_arrays(RunConfig())
