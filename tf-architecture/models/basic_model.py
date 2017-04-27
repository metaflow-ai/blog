import os, copy
import tensorflow as tf

class BasicAgent(object):
    def __init__(self, config, env):
        # I like to keep the best HP found so far inside the model itself
        if config['best']:
            config.update(self.get_best_config(config['env_name']))
            
        # I make a deepcopy of the configuration before using it
        # to avoid any potential mutation when I iterate asynchronously on configurations
        self.config = copy.deepcopy(config)

        if config['debug']:
            print('config', self.config)

        # When working with NN, one usually initialize randomly
        # You want to be able to reproduce it so make sure you store it
        # and use it in your TF graph (tf.set_random_seed() for example)
        self.random_seed = self.config['random_seed']

        # Then follows other global needed variables shared by all models
        self.result_dir = self.config['result_dir']
        self.max_iter = self.config['max_iter']
        self.lr = self.config['lr']
        self.nb_units = self.config['nb_units']
        # etc.
        
        # This call is made to allow children models to add their own properties
        # without having to override the __init__ function
        self.set_agent_props()

        # Each model should provide its own build_grap function obviously
        self.graph = self.build_graph(tf.Graph())

        # Any operations that should be in the that is shared by all models
        # can be added here
        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=50,
            )
            self.init_op = tf.global_variables_initializer()
        
        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        
        # This function is not always common to all models, that's why it's again
        # separated from the __init__ one
        self.init()

        # At the end of this function, you want your model to be ready to train/infer!

    def set_agent_props(self):
        # This function is here to be overrided completely.
        # When you look at your model, you want to know exactly which custom options it needs.
        pass

    def get_best_config(self):
        # This function is here to be overrided completely.
        # It returns a dictionnary used to update the initial configuration (see __init__)
        return {} 

    @staticmethod
    def get_random_config(fixed_params={}):
        # Why static? you want to be able to pass this function to other processes
        # so they can independently generate random configuration of this prcesise model
        raise Exception('The get_random_config function must be overrided by the agent')

    def build_graph(self, graph):
        raise Exception('The build_graph function must be overrided by the agent')

    def infer(self):
        raise Exception('The infer function must be overrided by the agent')

    def learn_from_epoch(self):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overrided by the agent')

    def train(self, save_every=1):
        # This function is usually common to all your models, Here is an example
        for epoch_id in range(0, self.max_iter):
            self.learn_from_epoch()

            # If you don't want to save during training, you can just pass a negative number
            if save_every > 0 and epoch_id % save_every == 0:
                self.save()

    def save(self):
        # This function is usually common to all your models, Here is an example
        global_step_t = tf.train.get_global_step(self.graph)
        global_step, episode_id = self.sess.run([global_step_t, self.episode_id])
        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.result_dir, global_step))
        self.saver.save(self.sess, self.result_dir + '/agent-ep_' + str(episode_id), global_step)

        # I always keep the configuration that
        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)


    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidded cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:

            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def infer(self):
        # This function is usually common to all your models
        # My inferring code
        