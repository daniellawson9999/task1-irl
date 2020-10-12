import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder
import numpy as np
import gym


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None,
                 sess=None,trainable_variance=True, trainable_bias=True, init_logstd=0, clip=None, scope_name='pi', **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01,
                                                    trainable_variance=trainable_variance,
                                                    trainable_bias=trainable_bias,
                                                    init_logstd=init_logstd,
                                                    clip=clip)

        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self.action = tf_util.switch(self.stochastic, self.pd.sample(), self.pd.mode())
        self.neglogp = self.pd.neglogp(self.action)
        self.logits=tf.nn.softmax(self.pd.flatparam())
        self.sess = sess
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]


    def _evaluate(self, variables, observation, stochastic, **extra_feed):
        sess = self.sess or tf.get_default_session()
        feed_dict = {self.X: adjust_shape(self.X, observation),
                     self.stochastic: stochastic}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, stochastic=False, logits=False, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp, logits = self._evaluate([self.action, self.vf, self.state, self.neglogp, self.logits], observation, stochastic, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, logits

    def get_weights(self, layer_wise=False):
        sess = self.sess or tf.get_default_session()
        if not layer_wise:
            layers = sess.run(self.vars)
            weights = []
            for layer in layers:
                weights.append(layer.ravel())
            return np.concatenate(weights)
        else:
            return sess.run(self.vars)

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_variables(save_path)#, sess=self.sess

    def load(self, load_path,extra_vars=None):
        tf_util.load_variables(load_path,extra_vars=extra_vars)#, sess=self.sess

def build_policy(env, policy_network, value_network=None, normalize_observations=False, estimate_q=False,
                 trainable_variance=True, trainable_bias=True, init_logstd=0,clip=None, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None, scope_name="pi"):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        #encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent, recurrent_tensors = policy_network(encoded_x)

            if recurrent_tensors is not None:
                # recurrent architecture, need a few more steps
                nenv = nbatch // nsteps
                assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                extra_tensors.update(recurrent_tensors)


        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                vf_latent, _ = _v_net(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            trainable_variance=trainable_variance,
            trainable_bias=trainable_bias,
            init_logstd=init_logstd,
            clip=clip,
            scope_name=scope_name,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

