# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Runs the learner/evaluator."""

import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from meshgraphnets import cfd_eval
from meshgraphnets import cfd_model
from meshgraphnets import NPS_model
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
from meshgraphnets import core_model
from meshgraphnets import dataset
# import horovod.tensorflow as hvd


FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', None, ['cfd', 'cloth', 'NPS'],
                  'Select model to run.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None,
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')
flags.DEFINE_integer('dim', 2, 'NPS dimension')
flags.DEFINE_integer('nfeat_in', 1, 'nfeat_in')
flags.DEFINE_integer('nfeat_out', -1, 'nfeat_out')
flags.DEFINE_integer('nfeat_latent', 128, 'nfeat_latent in GNN')
flags.DEFINE_integer('n_mpassing', 15, 'num. of message passing')
flags.DEFINE_integer('nlayer_mlp', 2, 'No. of layer in MLP')
flags.DEFINE_float('noise', -1.0, 'noise magnitude')
flags.DEFINE_integer('periodic', 0, 'NPS periodic boundary condition')
flags.DEFINE_integer('batch', 4, 'batch size')
flags.DEFINE_float('lr', 1e-4, 'learning rate')
flags.DEFINE_integer('lr_decay', 5000000, help='Learning rate decay.')
flags.DEFINE_boolean('rotate', False, help='Data augmentation by rotation')
flags.DEFINE_boolean('cache', False, help='Cache whole dataset into memory')
flags.DEFINE_boolean('randommesh', False, help='Data augmentation by generating random points and associated mesh')
# flags.DEFINE_float('random_lower', 0.3, 'ratio of selected points: lower bound')
# flags.DEFINE_float('random_upper', 0.8, 'ratio of selected points: upper bound')
# AMR options
flags.DEFINE_integer('amr_N', 64, 'system size, i.e. how many (fine) grids totally')
flags.DEFINE_integer('amr_N1', 1, 'how many (fine) grids to bin into one, 1 to disable')
flags.DEFINE_integer('amr_buffer', 1, 'how many buffer grids (must be 0 or 1)')
flags.DEFINE_float('amr_threshold', 1e-3, 'threshold to coarsen regions if values are close')


PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'NPS': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=NPS_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}


def learner(model, params, mesher=None):
  """Run a learner job."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')
  if FLAGS.cache:
    ds = ds.cache()
  if FLAGS.randommesh:
    ds = ds.map(dataset.augment_by_randommesh, periodic=FLAGS.periodic)
  if FLAGS.rotate:
    ds = ds.map(dataset.augment_by_rotation)
  ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
  ds = dataset.split_and_preprocess(ds, noise_field=params['field'],
                                    noise_scale=params['noise'] if FLAGS.noise<0 else FLAGS.noise,
                                    noise_gamma=params['gamma'])
  if mesher is not None:
    ds = dataset.remesh(ds, mesher, random_translate=False)
  ds = dataset.batch_dataset(ds, FLAGS.batch)
  inputs = tf.data.make_one_shot_iterator(ds).get_next()

  loss_op = model.loss(inputs)
  global_step = tf.train.create_global_step()
  lr = tf.train.exponential_decay(learning_rate=FLAGS.lr,
                                  global_step=global_step,
                                  decay_steps=FLAGS.lr_decay,
                                  decay_rate=0.1) + 1e-6
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  train_op = optimizer.minimize(loss_op, global_step=global_step)
  # Don't train for the first few steps, just accumulate normalization stats
  train_op = tf.cond(tf.less(global_step, 1000),
                     lambda: tf.group(tf.assign_add(global_step, 1)),
                     lambda: tf.group(train_op))

  with tf.train.MonitoredTrainingSession(
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.num_training_steps)],
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=600) as sess:

    while not sess.should_stop():
      _, step, loss = sess.run([train_op, global_step, loss_op])
      if step % 1000 == 0:
        logging.info('Step %d: Loss %g', step, loss)
    logging.info('Training complete.')
  evaluator(model, params, 'valid', None)


def evaluator(model, params, data_name, rollout_path, mesher=None):
  """Run a model rollout trajectory."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, data_name)
  ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  scalar_op, traj_ops = params['evaluator'].evaluate(model, inputs)
  tf.train.create_global_step()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None) as sess:
    trajectories = []
    scalars = []
    mse_list = []
    for traj_idx in range(FLAGS.num_rollouts):
      logging.info('Rollout trajectory %d', traj_idx)
      scalar_data, traj_data = sess.run([scalar_op, traj_ops])
      trajectories.append(traj_data)
      error = traj_data['pred_velocity'] - traj_data['gt_velocity']
      mse_list.append((error**2).mean(axis=1))
      scalars.append(scalar_data)
    for key in scalars[0]:
      logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
    print(f'RMSE   total {np.sqrt(np.mean(mse_list))}')
    print(f' per_channel {np.sqrt(np.mean(mse_list,axis=(0,1)))}')
    print(f'    per_step {np.sqrt(np.mean(mse_list,axis=(0,2)))}')
    print(f'    per_traj {np.sqrt(np.mean(mse_list,axis=(1,2)))}')
    if rollout_path:
      with open(rollout_path, 'wb') as fp:
        pickle.dump(trajectories, fp)


def main(argv):
  del argv
  tf.enable_resource_variables()
  tf.disable_eager_execution()
  params = PARAMETERS[FLAGS.model]
  nfeat_out = params['size'] if FLAGS.nfeat_out<0 else FLAGS.nfeat_out
  learned_model = core_model.EncodeProcessDecode(
      output_size=nfeat_out,
      latent_size=FLAGS.nfeat_latent,
      num_layers=FLAGS.nlayer_mlp,
      message_passing_steps=FLAGS.n_mpassing)
  if FLAGS.model in ['NPS']:
    model = params['model'].Model(learned_model, dim=FLAGS.dim, periodic=bool(FLAGS.periodic), nfeat_in=FLAGS.nfeat_in,
    nfeat_out=nfeat_out)
  else:
    model = params['model'].Model(learned_model)
  if FLAGS.amr_N1 > 1:
    from meshgraphnets import amr
    print('''************* WARNING *************
    The present AMR implementation assumes an input field on a cubic grid of size amr_N
    ordered naturally. Make sure your dataset follows this convention''')
    mesher = amr.amr_state_variables(FLAGS.dim, [FLAGS.amr_N]*FLAGS.dim,
      [FLAGS.amr_N//FLAGS.amr_N1]*FLAGS.dim,
      tf.zeros([FLAGS.amr_N**FLAGS.dim,1],dtype=tf.float32),
      refine_threshold=FLAGS.amr_threshold, buffer=FLAGS.amr_buffer)
  else:
    mesher = None
  if FLAGS.mode == 'train':
    learner(model, params, mesher)
  elif FLAGS.mode == 'eval':
    evaluator(model, params, FLAGS.rollout_split, FLAGS.rollout_path, mesher)

if __name__ == '__main__':
  app.run(main)
