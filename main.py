from build_dataset import build_embedding_dims_autoencoder
from _2vector import save_embedding_dims
from network import train_evaluate
import tick_tock
layers = [1, 2, 3, 4]
# 1 means one layer GRU
# 2 means two layers GRU
# 3 means one layer LSTM
# 4 means two layers LSTM
if __name__ == '__main__':
    dims = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    timer = tick_tock.Timer()
    timer.tick()
    for dim in dims:
        build_embedding_dims_autoencoder(dim)
        save_embedding_dims()
        for layer in layers:
            train_evaluate(dim, layer, training_size=1)
    timer.tock()
    print(timer.last_time())
