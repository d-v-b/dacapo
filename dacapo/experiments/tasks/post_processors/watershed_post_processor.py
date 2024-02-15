from pathlib import Path
from dacapo.blockwise.scheduler import segment_blockwise
from dacapo.compute_context import ComputeContext, LocalTorch
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray
from dacapo.store.array_store import LocalArrayIdentifier

from .watershed_post_processor_parameters import WatershedPostProcessorParameters
from .post_processor import PostProcessor

from funlib.geometry import Coordinate, Roi


import numpy as np

from typing import List


class WatershedPostProcessor(PostProcessor):
    def __init__(self, offsets: List[Coordinate]):
        self.offsets = offsets

    def enumerate_parameters(self):
        """Enumerate all possible parameters of this post-processor. Should
        return instances of ``PostProcessorParameters``."""

        for i, bias in enumerate([0.1, 0.25, 0.5, 0.75, 0.9]):
            yield WatershedPostProcessorParameters(id=i, bias=bias)

    def set_prediction(self, prediction_array_identifier):
        self.prediction_array = ZarrArray.open_from_array_identifier(
            prediction_array_identifier
        )

    def process(
        self,
        parameters: WatershedPostProcessorParameters,
        output_array_identifier: LocalArrayIdentifier,
        compute_context: ComputeContext | str = LocalTorch(),
        num_workers: int = 16,
        block_size: Coordinate = Coordinate((64, 64, 64)),
        context: Coordinate = Coordinate((32, 32, 32)),
    ):
        output_array = ZarrArray.create_from_array_identifier(
            output_array_identifier,
            [axis for axis in self.prediction_array.axes if axis != "c"],
            self.prediction_array.roi,
            None,
            self.prediction_array.voxel_size,
            np.uint64,
        )

        read_roi = Roi((0, 0, 0), self.prediction_array.voxel_size * block_size)
        # run blockwise prediction
        pars = {
            "offsets": self.offsets,
            "bias": parameters.bias,
        }
        segment_blockwise(
            segment_function_file=str(
                Path(Path(__file__).parent, "blockwise", "watershed_function.py")
            ),
            compute_context=compute_context,
            context=context,
            total_roi=self.prediction_array.roi,
            read_roi=read_roi.grow(context, context),
            write_roi=read_roi,
            num_workers=num_workers,
            max_retries=2,  # TODO: make this an option
            timeout=None,  # TODO: make this an option
            ######
            input_array_identifier=LocalArrayIdentifier(
                self.prediction_array.file_name, self.prediction_array.dataset
            ),
            output_array_identifier=output_array_identifier,
            parameters=pars,
        )

        return output_array
