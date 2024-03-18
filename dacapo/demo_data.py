from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
import os
import time
from typing import Any, Dict, List, Optional
import click
import numpy as np
import zarr
from zarr.storage import FSStore
import logging
from dataclasses import dataclass

@dataclass
class ScaleTranslation:
    scale: List[float]
    translation: List[float]

    def apply(self, value: List[float]) -> List[float]:
        scaled = [s * v for s,v in zip(self.scale, value)]
        translated = [t + v for t,v in zip(self.translation, scaled)]
        return translated 

collection_name = 'jrc_hela-2'
source_root = f's3://janelia-cosem-datasets/{collection_name}/{collection_name}.zarr'
# the path relative to the source root where the fibsem data is sourced
fibsem_path = 'recon-1/em/fibsem-uint8'

store_chunks = (64,) * 3

# each crop will be accompanied by fibsem data, which should be padded by this amount (in array elements)
fibsem_padding = (256,) * 3

# the default location where data will be saved
default_dest = f'./data/{collection_name}/'

# crop names should be joined to this path to resolve them to a zarr group
crop_path_template = 'recon-1/labels/groundtruth/'

# the default list of crop names to locally persist
default_crops = [
    "crop1",
    "crop3",
    "crop4",
    "crop6",
    "crop7",
    "crop8",
    "crop9",
    "crop13",
    "crop14",
    "crop15",
    "crop16",
    "crop18",
    "crop19",
    "crop23",
    "crop28",
    "crop54",
    "crop55",
    "crop56",
    "crop57",
    "crop58",
    "crop59",
    "crop94",
    "crop95",
    "crop96",
    "crop113",
    "crop155",
    ]

# for testing, reduce the number of crops

default_crops = default_crops[:2]

def get_scale_translations(ome_ngff_metadata: Dict[str, Any]) -> Dict[str, ScaleTranslation]:
    # represent arrays by name
    result = {}
    maybe_base_tx = ome_ngff_metadata.get('coordinateTransformations', None)
    
    if maybe_base_tx == [] or maybe_base_tx is None:
        base_tx = ScaleTranslation(scale=[1,1,1], translation=[0,0,0])
    else:
        if len(maybe_base_tx) == 1:
            base_tx = ScaleTranslation(scale=maybe_base_tx[0]["scale"], translation=[0,0,0])
        else:
            base_tx = ScaleTranslation(scale=maybe_base_tx[0]["scale"], translation=maybe_base_tx[1]["translation"])
    
    for dataset in ome_ngff_metadata["datasets"]:
        tx_meta = dataset["coordinateTransformations"]
        tx = ScaleTranslation(
            scale=[a * b for a, b in zip(tx_meta[0]["scale"], base_tx.scale)],
            translation=[a + b for a,b in zip(tx_meta[1]["translation"], base_tx.translation)]
            )
        result[dataset['path']] = tx
    return result

def prepare_group(source_root: str, source_path: str, dest_root: str, dest_path: str, pool: ThreadPoolExecutor, **kwargs) -> zarr.Group:
    """
    Ensure that a Zarr group stored at dest_root / dest_path has the same structure as the zarr group stored at 
    source_root / source_path. Returns the Zarr group at dest_root / dest_path.
    """
    source_group = zarr.open_group(FSStore(source_root), path=source_path, mode='r')   
    dest_group = zarr.open_group(FSStore(dest_root, dimension_separator='/'), path=dest_path)
    result_group = copy_group_structure(source_group=source_group, dest_group=dest_group, pool=pool, **kwargs)
    return result_group

def require_array(group: zarr.Group, key: str, value: zarr.Array, **kwargs):
    arr = group.require_dataset(
                name=key, 
                shape=value.shape, 
                dtype=value.dtype, 
                exact=True, 
                **kwargs)
    arr.attrs.update(value.attrs.asdict())
    return arr

def copy_group_structure(source_group: zarr.Group, dest_group: zarr.Group, pool: Optional[ThreadPoolExecutor], **kwargs):
    """
    Recursively copies the structure of `source_group` to `dest_group`, 
    where "structure" is defined as the the attributes + members (subgroups, and subarrays) of a group.
    
    This function does not copy an array data.
    
    Parameters
    ----------
    source_group: zarr.Group
        The group to copy structure from.
    dest_group: zarr.Group
        The group to copy structure to.
    **kwargs:
        Extra key-value arguments will be passed to `group.require_dataset`
    Returns
    -------
    zarr.Group
        dest_group comes out
    """
    dest_group.attrs.update(source_group.attrs.asdict())
    futures = []
    for key, value in source_group.items():
        if isinstance(value, zarr.Array):
            futures.append(pool.submit(require_array, dest_group, key, value, **kwargs))
        else:
            futures.append(pool.submit(lambda v: dest_group.require_group(name=v), key))
    results = wait(futures)
    for result_done in results.done:
        group_or_array = result_done.result()
        if isinstance(group_or_array, zarr.Group):
            copy_group_structure(
                source_group=source_group[group_or_array.name], 
                dest_group = group_or_array,
                pool = pool)

    return dest_group

def save_fibsem_region(
        dest_crop_group: zarr.Group,
        source_fibsem_group: zarr.Group, 
        dest_fibsem_group: zarr.Group
        ) -> zarr.Group:

    crop_scale_translate = get_scale_translations(dest_crop_group.attrs["multiscales"][0])
    fibsem_scale_translate = get_scale_translations(dest_fibsem_group.attrs["multiscales"][0])    

    crop_s0_shape = zarr.open_array(
        dest_crop_group.store, 
        path=os.path.join(dest_crop_group.path, 's0')).shape
    
    crop_s0_tx = crop_scale_translate['s0']
    
    fibsem_s0_tx = fibsem_scale_translate['s0']
    crop_s0_origin_idx = []
    
    for axis in range(len(crop_s0_shape)):
        crop_s0_origin_idx.append(crop_s0_tx.translation[axis] / fibsem_s0_tx.scale[axis])

    for name, tx in fibsem_scale_translate.items(): 
        ds_idx = int(name.lstrip('s'))
        interval = []
        
        source_array = zarr.open_array(
            store=source_fibsem_group.store, 
            path=os.path.join(source_fibsem_group.path, name))
        
        dest_array = zarr.open_array(
            store=dest_fibsem_group.store, 
            path=os.path.join(dest_fibsem_group.path, name))
        
        for idx, c, s, f in zip(range(len(crop_s0_shape)), crop_s0_origin_idx, crop_s0_shape, fibsem_padding):
            scale = fibsem_s0_tx.scale[idx] / crop_s0_tx.scale[idx]
            ds = (2 ** ds_idx)
            start = int((c - f)  / ds)
            stop = int(((s / scale) + c + f) / ds) 
            interval.append(slice(max(0, start), min(stop, source_array.shape[idx])))
        
        dest_array[tuple(interval)] = source_array[tuple(interval)]
    return dest_fibsem_group

def save_labels(
        source_crop_group: zarr.Group, 
        dest_crop_group: zarr.Group, 
) -> zarr.Group:
    crop_scale_translate = get_scale_translations(dest_crop_group.attrs["multiscales"][0])
    for name in crop_scale_translate.keys():
        arr_path = os.path.join(dest_crop_group.path, name)
        dest_arr = zarr.open_array(
            store=dest_crop_group.store, 
            path=arr_path,
            write_empty_chunks=False
            )
        do_write = True
        if "cellmap" in dest_arr.attrs:
            cellmap_meta = dest_arr.attrs["cellmap"]
            if "annotation" in cellmap_meta:
                # assume we are working with cellmap annotation metadata
                if cellmap_meta["annotation"]["complement_counts"]["absent"] == np.prod(dest_arr.shape):
                    do_write = False

        if do_write:
            dest_arr[:] = zarr.open_array(source_crop_group.store, path=arr_path)

    return dest_crop_group

@click.command('download-demo-data')
@click.option('-d', '--dest', type=click.STRING)
@click.option('-n', '--num-workers', type=click.INT, default=None)
def download_demo_data_cli(dest: str | None, num_workers: int | None) -> None:
    
    #logger = logging.getLogger(name=__name__)
    #logger.setLevel('INFO')
    #logger.addHandler(logging.StreamHandler())
    
    if dest is None:
        dest = default_dest
    
    pool = ThreadPoolExecutor(max_workers=num_workers)
    print(f'Using {pool._max_workers} workers for parallelism.')
    dest_root = os.path.join(default_dest, collection_name + '.zarr')
    start = time.time()
    print(f'Begin preparing the path {dest_root} to store FIB-SEM data...')
    dest_fibsem_group = prepare_group(
        source_root=source_root, 
        source_path=fibsem_path, 
        dest_root=dest_root, 
        dest_path=fibsem_path, 
        chunks=store_chunks,
        pool = pool)
    print(f'Completed preparing the path {dest_root} for FIB-SEM data.')
    
    print('Begin downloading ground truth crops and corresponding FIB-SEM data...')
    for crop in default_crops:
        crop_path = crop_path_template + crop
        print(f'Begin preparing storage for {crop} at {os.path.join(dest_root, crop_path)}.')        
        dest_crop_group = prepare_group(
            source_root=source_root, 
            source_path=crop_path,
            dest_root=dest_root,
            dest_path=crop_path,
            chunks=store_chunks,
            pool=pool)


        futures = []

        # submit computation for saving the FIB-SEM data
        futures.append(pool.submit(
            save_fibsem_region,
            dest_crop_group=dest_crop_group["all"],
            source_fibsem_group=zarr.open_group(source_root, path=fibsem_path),
            dest_fibsem_group=dest_fibsem_group
        ))
        
        for name, group in dest_crop_group.groups():
            fut = pool.submit(save_labels, 
                source_crop_group=zarr.open_group(
                    source_root, 
                    path=os.path.join(crop_path, name)),
                dest_crop_group=group,
            )
            futures.append(fut)
        for result in as_completed(futures):
            saved_group = result.result()
            group_name = saved_group.name
            print(f'Finished saving {saved_group.store.path}{group_name}')
        print(f'Completed downloading FIB-SEM data and labels for {crop}.')
    print(f'Completed saving all images after {time.time() - start}s.')