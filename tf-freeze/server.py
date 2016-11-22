import json, argparse, time

import tensorflow as tf
from load import load_graph

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        x_in = json.loads(params['x'])
    else:
        params = json.loads(data)
        x_in = params['x']
    print(params)
    
    y_out = persistent_sess.run(y, feed_dict={
        x: x_in
        # x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
    })
    json_data = json.dumps({'y': y_out.tolist()})

    print("Time spent handling the request: %f" % (time.time() - start))
    return json_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--gpu_memory", default=.2, type=float, help="GPU memory per process")
    args = parser.parse_args()


    print('Loading the model')
    graph = load_graph(args.frozen_model_filename)
    x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')

    print('Starting Session, setting the GPU memory usage to %f' % args.gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=graph, config=sess_config)

    print('Starting the API')
    app.run()
