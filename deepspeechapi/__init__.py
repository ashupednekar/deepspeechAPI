from deepspeech import Model

model_path = 'assets/200output_graph.pb'
alphabet_path = 'assets/alpha_small.txt'
ds = Model(model_path, 26, 9, alphabet_path, 500)
