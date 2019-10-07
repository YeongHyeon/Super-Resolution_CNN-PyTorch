import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch

import source.neuralnet as nn
import source.datamanager as dman
import source.solver as solver
import source.stamper as stamper
stamper.print_stamp()

def main():

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    srnet = nn.NeuralNet(device=device)

    dataset = dman.DataSet()

    solver.training(neuralnet=srnet, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch)
    solver.validation(neuralnet=srnet, dataset=dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5000, help='-')
    parser.add_argument('--batch', type=int, default=16, help='-')

    FLAGS, unparsed = parser.parse_known_args()

    main()
