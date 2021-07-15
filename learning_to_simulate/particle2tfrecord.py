import numpy as np
import scipy
import json
import os, itertools
import tensorflow.compat.v1 as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('traj', nargs='+', help='trajectory from e.g. LAMMPS')
parser.add_argument('--periodic', action='store_true', default=False, help='Periodic boundary condition')
parser.add_argument('-o', default='output', help='output tfrecord file')
args = parser.parse_args()


# import importlib.util
# spec = importlib.util.spec_from_file_location("MD_reader", "MD_reader.py")
# MD_reader = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(MD_reader)
import MD_reader

def traj2tfrecord(filenames_traj, fname, verbose=True, periodic=False, dt=0.01):
    """
    Converts a Numpy array a tfrecord file.
    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            # return tf.train.Feature(float_list=tf.train.FloatList(value=ndarray.flatten()))
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[ndarray.tobytes()]))
        elif dtype_ == np.int64 or dtype_ == np.int32:
            # return tf.train.Feature(int64_list=tf.train.Int64List(value=ndarray.flatten()))
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[ndarray.tobytes()]))
        else:  
            raise ValueError("The input should be numpy ndarray but got {}".format(ndarray.dtype))
            

    # write tfrecord
    f_out = fname + '.tfrecord'
    writer = tf.python_io.TFRecordWriter(f_out)
    for dump_file in filenames_traj:
        print(f'Processing {dump_file}')
        pbc, particle_type, lattices, position = MD_reader.read_dump(dump_file, write=False)
        particle_type = particle_type.astype(np.int64)
        lattices = lattices.astype(np.float32)
        position = position.astype(np.float32)
        # for pos in trajectories:
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                # 'key': 0,
                'particle_type': _dtype_feature(particle_type)
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'position': tf.train.FeatureList(feature=[_dtype_feature(position
                )])
            })
        )                
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

    # write meta file 
    meta_dict = {'sequence_length': len(position)-1, 
        "default_connectivity_radius": 2.6, "dim": position.shape[-1], "dt": dt}
    meta_dict['bounds'] = lattices[0].tolist()
    meta_dict['vel_mean'] = [0,0,0]
    meta_dict['vel_std'] = [1,1,1]
    meta_dict['acc_mean'] = [0,0,0]
    meta_dict['acc_std'] = [1,1,1]
    with open(fname + ".json", "w") as outfile:
        json.dump(meta_dict, outfile)
    if verbose:
        print(f"Writing {f_out} done!")

traj2tfrecord(args.traj, args.o, verbose=True, periodic=args.periodic)
