import numpy as np
import tensorflow as tf

cnt=0
def load_pb(path_to_pb, input):
    input = tf.transpose(input ,[0, 3, 1, 2])
    input = input /255
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    #with tf.get_default_graph() as graph:
    global cnt
    cnt+=1
    output, = tf.import_graph_def(graph_def, name='model%d'%cnt, input_map={
                        "input: 0": input, }, return_elements=['add_16: 0'])
    #print(output.shape.as_list())
    return tf.get_default_graph(), output

def get_model(input, ):
    graph, output = load_pb("./pretrained/model_cifar_wrn.pb", input)
    #print([n.name for n in tf.get_default_graph().as_graph_def().node])
    #output_tensor = graph.get_tensor_by_name('model/add_16: 0')
    #input_tensor = graph.get_tensor_by_name('model/input:0')
    return output#input_tensor, output_tensor


class container:
    def __init__(self):
        pass

def build_model(x,y, conf=1):

    cont = container()
    logits = get_model(x)
    cont.logits = logits

    predictions = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predictions, y)

    cont.correct_prediction= correct_prediction
    cont.accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32))

    with tf.variable_scope('costs'):
        cont.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=y)

        label_one_hot = tf.one_hot(y, depth=10)
        wrong_logit = tf.reduce_max(
            logits * (1-label_one_hot) - label_one_hot * 1e7, axis=-1)
        true_logit = tf.reduce_sum(
            logits * label_one_hot, axis=-1)
        cont.target_loss = - \
            tf.reduce_sum(tf.nn.relu(true_logit - wrong_logit + conf))

        cont.xent = tf.reduce_sum(logits, name='y_xent')
        cont.mean_xent = tf.reduce_mean(logits)
    return cont

def gen_pb():
    from trades.models.wideresnet import WideResNet
    import torch    
    import onnx
    from onnx_tf.backend import prepare

    device = torch.device("cuda")
    model = WideResNet().to(device)
    model.load_state_dict(torch.load('./model_cifar_wrn.pt'))
    model.eval()

    dummy_input = torch.from_numpy(
        np.zeros((64, 3, 32, 32),)).float().to(device)
    dummy_output = model(dummy_input)

    torch.onnx.export(model, dummy_input, './model_cifar_wrn.onnx',
                      input_names=['input'], output_names=['output'])

    model_onnx = onnx.load('./model_cifar_wrn.onnx')

    tf_rep = prepare(model_onnx)

    # Print out tensors and placeholders in model (helpful during inference in TensorFlow)
    print(tf_rep.tensor_dict)

    # Export model as .pb file
    tf_rep.export_graph('./model_cifar_wrn.pb')

if __name__=="__main__":
    gen_pb()
    """
    input_tensor, output_tensor = get_model()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    output_val = sess.run(output_tensor, feed_dict={
                          input_tensor: np.zeros((64, 3, 32, 32),)})
    print(output_val)
    """