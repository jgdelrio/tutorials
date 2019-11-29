#
# Tensorflow Tools
#
import os
from tensorflow import saved_model, train, graph_util
from tensorflow.python.tools import freeze_graph


def save_tf_model(self, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    saved_model.save(directory)
    return directory


def save_tf1(self, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename + '.ckpt')
    self.saver.save(self.sess, filepath)
    return filepath


def save_tf1_as_pb(self, directory, filename, method=1):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save check point for graph frozen later
    ckpt_filepath = self.save(directory=directory, filename=filename)
    pbtxt_filename = filename + '.pbtxt'
    pbtxt_filepath = os.path.join(directory, pbtxt_filename)
    pb_filepath = os.path.join(directory, filename + '.pb')
    # This only saves the graph, not the variables. We have to freeze the model 1st.
    train.write_graph(graph_or_graph_def=self.sess.graph_def, logdir=directory, name=pbtxt_filename, as_text=True)

    # Freeze graph
    if method == 1:
        freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='', input_binary=False,
                                  input_checkpoint=ckpt_filepath, output_node_names='cnn/output',
                                  restore_op_name='save/restore_all', filename_tensor_name='save/Const:0',
                                  output_graph=pb_filepath, clear_devices=True, initializer_nodes='')
    elif method == 2:
        from tensorflow import get_default_graph, gfile
        graph = get_default_graph()
        input_graph_def = graph.as_graph_def()
        output_node_names = ['cnn/output']

        output_graph_def = graph_util.convert_variables_to_constants(self.sess, input_graph_def, output_node_names)

        with gfile.GFile(pb_filepath, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
    else:
        raise ValueError("Unsupported method!")

    return pb_filepath