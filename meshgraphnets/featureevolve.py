import collections
import functools
import sonnet as snt
import tensorflow.compat.v1 as tf

from meshgraphnets.core_model import EncodeProcessDecode, GraphNetBlock

class FeatureEvolveDecode(EncodeProcessDecode):
  def __init__(self, *args, name='FeatureEvolveDecode', evolve_steps=3, **kwargs):
    super().__init__(*args, **kwargs, name=name)
    self._evolve_steps = evolve_steps

  # separate the message passing steps into two parts: process and evolve
  def _process(self, latent_graph):
    model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
    for _ in range(self._message_passing_steps):
      latent_graph = GraphNetBlock(model_fn, name='process')(latent_graph)
    return latent_graph

  @snt.reuse_variables
  def evolve(self, latent_graph):
    model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
    for _ in range(self._evolve_steps):
      latent_graph = GraphNetBlock(model_fn, name='evolve')(latent_graph)
    return latent_graph

  @snt.reuse_variables
  def featurize(self, graph):
    latent_graph = self._encoder(graph)
    latent_graph = self._process(latent_graph)
    return latent_graph

  @snt.reuse_variables
  def decoder(self, x):
    return self._decoder(x)

  # @snt.reuse_variables
  def _build(self, graph):
    latent_graph = self.featurize(graph)
    latent_graph = self.evolve(latent_graph)
    return self.decoder(latent_graph)

  @snt.reuse_variables
  def step(self, latent_graph):
    latent_graph = self.evolve(latent_graph)
    return latent_graph, self.decoder(latent_graph)
