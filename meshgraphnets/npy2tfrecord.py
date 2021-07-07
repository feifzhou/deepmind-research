import numpy as np
import scipy
import json
import os, itertools
import tensorflow.compat.v1 as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('npy', default=None, help='array')
parser.add_argument('--periodic', action='store_true', default=False, help='Periodic boundary condition')
parser.add_argument('-o', default='output', help='output tfrecord file')
args = parser.parse_args()


def cell2graph(a, cell_len, pbc=False):
    shape = a.shape[1:-1]
    dim = len(shape)
    xyz = [np.arange(shape[i]) for i,l in enumerate(cell_len)]
    mesh_pos = np.stack(np.meshgrid(*xyz), axis=-1).reshape((-1, dim))
    node_type = np.zeros_like(mesh_pos[:,:1], dtype='int32')
    field = a.reshape((a.shape[0], -1, a.shape[-1])).astype('float32')
    # print(f'debug dim {dim} mesh {mesh_pos} input a {a.shape}')
    if dim == 2:
        if pbc:
            corner = mesh_pos
        else:
            corner = np.stack(np.meshgrid(*[np.arange(shape[i]-1) for i,l in enumerate(cell_len)]), axis=-1).reshape((-1, dim))
        partitions = np.array([[[[0,0],[0,1],[1,0]],[[1,1],[0,1],[1,0]]], [[[0,0],[0,1],[1,1]],[[0,0],[1,0],[1,1]]]]) #itertools.combinations([[0,0],[0,1],[1,0],[1,1]], 3)
        cells = np.array([ij + partitions[np.random.choice(len(partitions))] for ij in corner[:,None,None,:]]).reshape((-1, 3, dim))
        cells = np.mod(cells, shape)
        cells = np.dot(cells, [1, shape[0]]).astype('int32')
    mesh_pos = (mesh_pos/(np.array(cell_len)[None,:])).astype('float32')
    # print(f'debug cells {cells.shape} mesh {mesh_pos.shape} node {node_type.shape} field {field.shape}')
    return cells, mesh_pos, node_type, field

def arr2tfrecord(arr, fname, verbose=True, dim=-1, cell_len=-1, periodic=False):
    """
    Converts a Numpy array a tfrecord file.
    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            # return tf.train.Feature(float_list=tf.train.FloatList(value=ndarray.flatten()))
            # print(f'debug', ndarray.tobytes())
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[ndarray.tobytes()]))
        elif dtype_ == np.int64 or dtype_ == np.int32:
            # return tf.train.Feature(int64_list=tf.train.Int64List(value=ndarray.flatten()))
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[ndarray.tobytes()]))
        else:  
            raise ValueError("The input should be numpy ndarray but got {}".format(ndarray.dtype))
            
    # prepare
    f_out = fname + '.tfrecord'
    arr = np.array(arr)
    nfeat = arr.shape[-1]
    trajectory_length = arr.shape[1]
    if dim == -1:
        dim = arr.ndim - 3
    assert (dim>=1) and (dim<=3), f'Invalid dimension {dim}'
    # length of the simulation cell
    if cell_len == -1:
        cell_len = arr.shape[2:2+dim]
    if arr.ndim == dim + 3:
        pass
    elif arr.ndim == dim + 2:
        arr = arr[..., None]
    else:
        raise f'ERROR input array should be {dim+3} dimensional'

    # write meta file
    meta_dict = {'trajectory_length': trajectory_length}
    meta_dict["field_names"] = [
        "cells",
        "mesh_pos",
        "node_type",
        "velocity"
    ]
    meta_dict["features"] = {
        "cells": {
            "type": "static",
            "shape": [1, -1, dim+1],
            "dtype": "int32"
        },
        "mesh_pos": {
            "type": "static",
            "shape": [1, -1, dim],
            "dtype": "float32"
        },
        "node_type": {
            "type": "static",
            "shape": [1, -1, 1],
            "dtype": "int32"
        },
        "velocity": {
            "type": "dynamic",
            "shape": [trajectory_length, -1, nfeat],
            "dtype": "float32"
        }
    }
    with open(fname + ".json", "w") as outfile:
        json.dump(meta_dict, outfile)

    # write tfrecord
    writer = tf.python_io.TFRecordWriter(f_out)
    for a in arr:
        cells, mesh_pos, node_type, field = cell2graph(a, cell_len, pbc=periodic)
        example = tf.train.Example(features=tf.train.Features(feature={
            "cells": _dtype_feature(cells),
            "mesh_pos": _dtype_feature(mesh_pos),
            "node_type": _dtype_feature(node_type),
            "velocity": _dtype_feature(field)
        }))
               
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()
    if verbose:
        print(f"Writing {f_out} done!")

arr2tfrecord(np.load(args.npy), args.o, verbose=True, cell_len=-1, periodic=False)
