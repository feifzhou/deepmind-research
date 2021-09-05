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
"""Plots a CFD trajectory rollout."""

import pickle

from absl import app
from absl import flags
import matplotlib
from matplotlib import animation
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('rollout_path', None, 'Path to rollout pickle file')
flags.DEFINE_integer('skip', 1, 'skip timesteps between animation')
flags.DEFINE_boolean('mirrory', False, 'Make mirror plots to better see periodic boundary condition along y')
flags.DEFINE_boolean('label', True, 'show label')
flags.DEFINE_boolean('mesh', False, 'show mesh')
flags.DEFINE_boolean('tri', False, 'use specified triangulation (--notri: calculate Delaunay triangulation) Do NOT set unless trigular mesh')
flags.DEFINE_float('scale', 1.0, 'image scale WRT default')
flags.DEFINE_string('cmap', 'viridis', 'color map')
flags.DEFINE_string('o', '', 'save as gif')

def run_animation(anim, fig):
    anim_running = True

    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    fig.canvas.mpl_connect('button_press_event', onClick)

def main(unused_argv):
  if FLAGS.o: matplotlib.use('Agg')
  # matplotlib.rc('image', cmap=plt.get_cmap(FLAGS.cmap))
  plt.rcParams['image.cmap'] = FLAGS.cmap
  with open(FLAGS.rollout_path, 'rb') as fp:
    rollout_data = pickle.load(fp)

  if FLAGS.mirrory:
    fig, axs = plt.subplots(2, 2, figsize=(16*FLAGS.scale, 16*FLAGS.scale))
  else:
    fig, axs = plt.subplots(1, 2, figsize=(16*FLAGS.scale, 8*FLAGS.scale))
  axs = axs.flatten()
  plt.subplots_adjust(0,0,0.95, 0.95, -0.08, -0.08)
  skip = FLAGS.skip
  num_steps = rollout_data[0]['gt_velocity'].shape[0]
  num_frames_per_rollout = (num_steps-1) // skip + 1
  num_frames = len(rollout_data) * num_frames_per_rollout

  # compute bounds
  bounds = []
  for trajectory in rollout_data:
    bb_min = trajectory['gt_velocity'].min(axis=(0, 1))
    bb_max = trajectory['gt_velocity'].max(axis=(0, 1))
    bounds.append((bb_min, bb_max))

  def remove_boundary_face(faces, pos, cutoff=2.0):
    face_diameter = np.stack([np.linalg.norm(pos[faces[:,ij[0]]]-pos[faces[:,ij[1]]],axis=1) for ij in ((0,1),(1,2),(0,2))], -1)
    face_diameter = np.max(face_diameter, axis=-1)
    return faces[np.where(face_diameter<cutoff)]

  def animate(num):
    step = (num%num_frames_per_rollout)*skip
    traj = num//num_frames_per_rollout
    for i, ax in enumerate(axs):
      col = i%2
      ax.cla()
      ax.set_aspect('equal')
      ax.set_axis_off()
      vmin, vmax = bounds[traj]
      pos = rollout_data[traj]['mesh_pos'][step]
      if FLAGS.tri:
        faces = rollout_data[traj]['faces'][step]
        faces = remove_boundary_face(faces, pos)
      else:
        faces = None
      velocity = rollout_data[traj][['gt_velocity','pred_velocity'][col]][step]
      triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
      ax.tripcolor(triang, velocity[:, 0], vmin=vmin[0], vmax=vmax[0])
      if FLAGS.mesh: ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
      if FLAGS.label:
        ax.set_title('%s traj %d step %d' % (['GT','PD'][col], traj, step))
    return fig,

  anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
  run_animation(anim, fig)
  if FLAGS.o:
      anim.save(FLAGS.o, writer='imagemagick', fps=6)
  else:
      plt.show(block=True)


if __name__ == '__main__':
  app.run(main)
