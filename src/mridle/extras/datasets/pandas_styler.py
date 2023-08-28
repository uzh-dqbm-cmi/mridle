from pathlib import PurePosixPath

from kedro.io.core import (
    AbstractVersionedDataSet,
    get_filepath_str,
    get_protocol_and_path,
    DataSetError,
    Version,
)

import fsspec
import numpy as np

from typing import Any, Dict


class PandasStylerHtml(AbstractVersionedDataSet):
    def __init__(self, filepath: str, version: Version = None):
        """Creates a new instance to load / save image data for given filepath.

        Args:
            filepath: The location of the table file to load / save data.
        """
        # parse the path and protocol (e.g. file, http, s3, etc.)
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)
        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    def _load(self) -> np.ndarray:
        raise DataSetError("`load` is not supported on PandasStylerHtml")

    def _save(self, data) -> None:
        """Saves a pandas table as html to the specified filepath."""
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        df = data.data  # Extract the DataFrame from the styler
        df_html = df.to_html()  # Convert DataFrame to HTML
        with open(save_path, "w") as f:
            f.write(df_html)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)
