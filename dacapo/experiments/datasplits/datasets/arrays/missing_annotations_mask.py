from .array import Array

from funlib.geometry import Coordinate, Roi

from fibsem_tools.metadata.groundtruth import Label, LabelList

import neuroglancer

import numpy as np


class MissingAnnotationsMask(Array):
    """
    This is wrapper around a ZarrArray containing uint annotations.
    Complementary to the BinarizeArray class where we convert labels
    into individual channels for training, we may find crops where a
    specific label is present, but not annotated. In that case you
    might want to avoid training specific channels for specific
    training volumes.
    See package fibsem_tools for appropriate metadata format for indicating
    presence of labels in your ground truth.
    "https://github.com/janelia-cosem/fibsem-tools"
    """

    def __init__(self, array_config):
        self.name = array_config.name
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

        assert (
            "c" not in self._source_array.axes
        ), f"Cannot initialize a BinarizeArray with a source array with channels"

        self._groupings = array_config.groupings

    @property
    def axes(self):
        return ["c"] + self._source_array.axes

    @property
    def dims(self) -> int:
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
        return self._source_array.voxel_size

    @property
    def roi(self) -> Roi:
        return self._source_array.roi

    @property
    def writable(self) -> bool:
        return False

    @property
    def dtype(self):
        return np.uint8

    @property
    def num_channels(self) -> int:
        return len(self._groupings)

    @property
    def data(self):
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    @property
    def attrs(self):
        return self._source_array.attrs

    @property
    def channels(self):
        return (name for name, _ in self._groupings)

    def __getitem__(self, roi: Roi) -> np.ndarray:
        labels_list = LabelList.parse_obj({"labels": self.attrs["labels"]}).labels
        present_not_annotated = set(
            [
                label.value
                for label in labels_list
                if label.annotationState.present and not label.annotationState.annotated
            ]
        )
        labels = self._source_array[roi]
        grouped = np.ones((len(self._groupings), *labels.shape), dtype=np.bool)
        for i, (_, ids) in enumerate(self._groupings):
            if any([id in present_not_annotated for id in ids]):
                grouped[i] = False
            else:
                grouped[i] = labels > 0
        return grouped

    def _can_neuroglance(self):
        return self._source_array._can_neuroglance()

    def _neuroglancer_source(self):
        return self._source_array._neuroglancer_source()

    def _neuroglancer_layer(self):
        # Generates an Segmentation layer

        layer = neuroglancer.SegmentationLayer(source=self._neuroglancer_source())
        kwargs = {
            "visible": False,
        }
        return layer, kwargs

    def _source_name(self):
        return self._source_array._source_name()