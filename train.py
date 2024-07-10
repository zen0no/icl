import pyrallis
import os

# This lines removes non-determenistic behaviour of LSTM on CUDA
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


class TrainConfig:
    pass


@pyrallis.wrap()
def main(config: TrainConfig):
    pass

if __name__ == '__main__':
    main()