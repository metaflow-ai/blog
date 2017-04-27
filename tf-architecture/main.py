import os, json
import tensorflow as tf
import numpy as np

from models import make_models
from hpsearch import hyperband, randomsearch

# I personnaly like to always make my paths absolute
dir = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags

# Hyper-parameters search
flags.DEFINE_boolean('fullsearch', False, 'Perform a full search of hyperparameter space (hyperband -> lr search -> hyperband with best lr)')
flags.DEFINE_string('fixed_params', "{}", 'JSON inputs to fix some params in a HP search, ex: \'{"lr": 0.001}\'')
flags.DEFINE_boolean('dry_run', False, 'Perform a hyperband dry_run')
flags.DEFINE_integer('nb_process', 4, 'Number of parallel process to perform a hyperband search')

# Agent
flags.DEFINE_string('model_name', 'DQNAgent', 'Unique name of the model')
flags.DEFINE_boolean('best', False, 'Force to use the best known configuration')
flags.DEFINE_float('initial_mean', 0., 'Initial mean for NN')
flags.DEFINE_float('initial_stddev', 1e-2, 'Initial standard deviation for NN')
flags.DEFINE_float('lr', 1e-3, 'The learning rate of SGD')
flags.DEFINE_float('nb_units', 20, 'Number of hidden units in Deep learning agents')

# Environment
flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_integer('max_iter', 2000, 'Number of training step')
flags.DEFINE_string('result_dir', dir + '/results/' + flags.FLAGS.env_name + '/' + flags.FLAGS.agent_name + '/' + str(int(time.time())), 'Name of the directory to store/log the agent (if it exists, the agent will be loaded from it)')
flags.DEFINE_boolean('infer', False, 'Load an agent for playing')
flags.DEFINE_integer('random_seed', random.randint(0, sys.maxsize), 'Value of random seed')

def main(_):
    config = flags.FLAGS.__flags.copy()
    config["fixed_params"] = json.loads(config["fixed_params"])

    if config['fullsearch']:
        # My code for hyper-parameters search
    else:
        model = make_model(config)

        if config['infer']:
            # My code for inference
        else:
            # My code for training given a set of HP


if __name__ == '__main__':
  tf.app.run()