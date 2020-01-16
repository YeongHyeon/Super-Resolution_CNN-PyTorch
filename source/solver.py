import os, inspect, time

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def makedir(path):

    try: os.mkdir(path)
    except: pass

def save_graph(contents, xlabel, ylabel, savename):

    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()

def psnr(input, target):

    psnr = torch.log(1 / torch.sqrt(torch.mean((target - input)**2))) / np.log(10.0) * 20
    return psnr

def torch2npy(input):

    output = input.detach().numpy()
    return output

def training(neuralnet, dataset, epochs, batch_size):

    start_time = time.time()
    loss_tr = 0
    list_loss = []
    list_psnr = []
    list_psnr_static = []

    makedir(PACK_PATH+"/training")
    makedir(PACK_PATH+"/static")
    makedir(PACK_PATH+"/static/reconstruction")

    print("\nTraining SRCNN to %d epochs" %(epochs))

    X_static, Y_static, X_static_t, Y_static_t, _ = dataset.next_train(batch_size=1)
    img_input = np.squeeze(X_static, axis=0)
    img_ground = np.squeeze(Y_static, axis=0)
    plt.imsave("%s/static/bicubic.png" %(PACK_PATH), img_input)
    plt.imsave("%s/static/high-resolution.png" %(PACK_PATH), img_ground)

    writer = SummaryWriter()
    iteration = 0
    for epoch in range(epochs):

        while(True):
            X_tr, Y_tr, X_tr_t, Y_tr_t, terminator = dataset.next_train(batch_size=batch_size)

            img_recon = neuralnet.model(X_tr_t.to(neuralnet.device))
            mse = neuralnet.mse(input=img_recon.to(neuralnet.device), target=Y_tr_t.to(neuralnet.device))
            neuralnet.optimizer.zero_grad()
            mse.backward()
            neuralnet.optimizer.step()

            loss_tr, psnr_tr = mse.item(), psnr(input=img_recon.to(neuralnet.device), target=Y_tr_t.to(neuralnet.device)).item()
            list_loss.append(loss_tr)
            list_psnr.append(psnr_tr)

            writer.add_scalar('Loss/mse', loss_tr, iteration)
            writer.add_scalar('Loss/psnr', psnr_tr, iteration)

            iteration += 1
            if(terminator): break

        X_tmp, Y_tmp = np.expand_dims(X_tr_t[0], axis=0), np.expand_dims(Y_tr_t[0], axis=0)
        X_tmp_t, Y_tmp_t = torch.from_numpy(X_tmp), torch.from_numpy(Y_tmp)

        img_recon = neuralnet.model(X_tmp_t.to(neuralnet.device))
        tmp_psnr = psnr(input=img_recon.to(neuralnet.device), target=Y_tmp_t.to(neuralnet.device)).item()

        X_tmp = np.transpose(X_tmp, (0, 2, 3, 1))
        Y_tmp = np.transpose(Y_tmp, (0, 2, 3, 1))
        img_recon = np.transpose(torch2npy(img_recon.cpu()), (0, 2, 3, 1))

        img_input, img_recon, img_ground = np.squeeze(X_tmp, axis=0), np.squeeze(img_recon, axis=0), np.squeeze(Y_tmp, axis=0)

        plt.clf()
        plt.rcParams['font.size'] = 100
        plt.figure(figsize=(100, 40))
        plt.subplot(131)
        plt.title("Low-Resolution")
        plt.imshow(img_input)
        plt.subplot(132)
        plt.title("Reconstruction")
        plt.imshow(img_recon)
        plt.subplot(133)
        plt.title("High-Resolution")
        plt.imshow(img_ground)
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.savefig("%s/training/%09d_psnr_%d.png" %(PACK_PATH, epoch, int(tmp_psnr)))
        plt.close()

        """static img(test)"""
        img_recon = neuralnet.model(X_static_t.to(neuralnet.device))
        tmp_psnr = psnr(input=img_recon.to(neuralnet.device), target=Y_static_t.to(neuralnet.device)).item()
        img_recon = np.transpose(torch2npy(img_recon.cpu()), (0, 2, 3, 1))

        list_psnr_static.append(tmp_psnr)
        img_recon = np.squeeze(img_recon, axis=0)
        plt.imsave("%s/static/reconstruction/%09d_psnr_%d.png" %(PACK_PATH, epoch, int(tmp_psnr)), img_recon)

        print("Epoch [%d / %d] | Loss: %f  PSNR: %f" %(epoch, epochs, loss_tr, psnr_tr))
        torch.save(neuralnet.model.state_dict(), PACK_PATH+"/runs/params")

    print("Final Epcoh | Loss: %f  PSNR: %f" %(loss_tr, psnr_tr))

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    save_graph(contents=list_loss, xlabel="Iteration", ylabel="L2 loss", savename="loss")
    save_graph(contents=list_psnr, xlabel="Iteration", ylabel="PSNR (dB)", savename="psnr")
    save_graph(contents=list_psnr_static, xlabel="Iteration", ylabel="PSNR (dB)", savename="psnr_static")

def validation(neuralnet, dataset):

    if(os.path.exists(PACK_PATH+"/runs/params")):
        neuralnet.model.load_state_dict(torch.load(PACK_PATH+"/runs/params"))
        neuralnet.model.eval()

    makedir(PACK_PATH+"/test")
    makedir(PACK_PATH+"/test/reconstruction")

    start_time = time.time()
    print("\nValidation")
    for tidx in range(dataset.amount_te):

        X_te, Y_te, X_te_t, Y_te_t = dataset.next_test()
        if(X_te is None): break

        img_recon = neuralnet.model(X_te_t.to(neuralnet.device))
        tmp_psnr = psnr(input=img_recon.to(neuralnet.device), target=Y_te_t.to(neuralnet.device)).item()
        img_recon = np.transpose(torch2npy(img_recon.cpu()), (0, 2, 3, 1))

        img_recon = np.squeeze(img_recon, axis=0)
        plt.imsave("%s/test/reconstruction/%09d_psnr_%d.png" %(PACK_PATH, tidx, int(tmp_psnr)), img_recon)

        img_input = np.squeeze(X_te, axis=0)
        img_ground = np.squeeze(Y_te, axis=0)
        plt.imsave("%s/test/bicubic.png" %(PACK_PATH), img_input)
        plt.imsave("%s/test/high-resolution.png" %(PACK_PATH), img_ground)

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))
