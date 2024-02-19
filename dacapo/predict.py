from pathlib import Path

from dacapo.blockwise import run_blockwise
from dacapo.experiments import Run
from dacapo.store.create_store import create_config_store
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray

from funlib.geometry import Coordinate, Roi
import numpy as np
import zarr

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def predict(
    run_name: str,
    iteration: int,
    input_container: Path | str,
    input_dataset: str,
    output_path: Path | str,
    output_roi: Optional[Roi | str] = None,
    num_workers: int = 30,
    output_dtype: np.dtype | str = np.uint8,  # type: ignore
    overwrite: bool = True,
):
    """_summary_

    Args:
        run_name (str): _description_
        iteration (int): _description_
        input_container (Path | str): _description_
        input_dataset (str): _description_
        output_path (Path | str): _description_
        output_roi (Optional[str], optional): Defaults to None. If output roi is None,
            it will be set to the raw roi.
        num_workers (int, optional): _description_. Defaults to 30.
        output_dtype (np.dtype | str, optional): _description_. Defaults to np.uint8.
        overwrite (bool, optional): _description_. Defaults to True.
    """
    # retrieving run
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # get arrays
    raw_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    raw_array = ZarrArray.open_from_array_identifier(raw_array_identifier)
    output_container = Path(
        output_path,
        "".join(Path(input_container).name.split(".")[:-1]) + ".zarr",
    )  # TODO: zarr hardcoded
    prediction_array_identifier = LocalArrayIdentifier(
        output_container, f"prediction_{run_name}_{iteration}"
    )

    if output_roi is None:
        output_roi = raw_array.roi
    elif isinstance(output_roi, str):
        start, end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in output_roi.strip("[]").split(",")
            ]
        )
        output_roi = Roi(
            Coordinate(start),
            Coordinate(end) - Coordinate(start),
        )
        output_roi = output_roi.snap_to_grid(
            raw_array.voxel_size, mode="grow"
        ).intersect(raw_array.roi)

    if isinstance(output_dtype, str):
        output_dtype = np.dtype(output_dtype)

    model = run.model.eval()

    # get the model's input and output size

    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)
    input_shape = Coordinate(model.eval_input_shape)
    input_size = input_voxel_size * input_shape
    output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]

    logger.info(
        "Predicting with input size %s, output size %s", input_size, output_size
    )

    # calculate input and output rois

    context = (input_size - output_size) / 2
    _input_roi = output_roi.grow(context, context)

    logger.info("Total input ROI: %s, output ROI: %s", _input_roi, output_roi)

    # prepare prediction dataset
    axes = ["c"] + [axis for axis in raw_array.axes if axis != "c"]
    ZarrArray.create_from_array_identifier(
        prediction_array_identifier,
        axes,
        output_roi,
        model.num_out_channels,
        output_voxel_size,
        output_dtype,
        overwrite=overwrite,
    )

    # run blockwise prediction
    worker_file = str(Path(Path(__file__).parent, "blockwise", "predict_worker.py"))
    logger.info("Running blockwise prediction with worker_file: ", worker_file)
    run_blockwise(
        worker_file=worker_file,
        total_roi=_input_roi,
        read_roi=Roi((0, 0, 0), input_size),
        write_roi=Roi((0, 0, 0), output_size),
        num_workers=num_workers,
        max_retries=2,  # TODO: make this an option
        timeout=None,  # TODO: make this an option
        ######
        run_name=run_name,
        iteration=iteration,
        raw_array_identifier=raw_array_identifier,
        prediction_array_identifier=prediction_array_identifier,
    )

    container = zarr.open(str(prediction_array_identifier.container))
    dataset = container[prediction_array_identifier.dataset]
    dataset.attrs["axes"] = (  # type: ignore
        raw_array.axes if "c" in raw_array.axes else ["c"] + raw_array.axes
    )
