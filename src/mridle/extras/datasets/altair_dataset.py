from pathlib import PurePosixPath

from kedro.io.core import (
    AbstractVersionedDataSet,
    get_filepath_str,
    get_protocol_and_path,
    DataSetError,
)

import fsspec
import numpy as np

import altair as alt


class AltairDataSet(AbstractVersionedDataSet):
    def __init__(self, filepath: str):
        """Creates a new instance of ImageDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        # parse the path and protocol (e.g. file, http, s3, etc.)
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> np.ndarray:
        raise DataSetError("`load` is not supported on AltairDataSet")

    def _save(self, data: alt.Chart) -> None:
        """Saves an altair chart to the specified filepath."""
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        with self._fs.open(save_path, "wb") as fs_file:
            data.save(fs_file)
