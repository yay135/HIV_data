import datetime
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from feature_label_preprocessing_1 import DataPreprocessing
from logger import log_to_file, setup_logging
from lstm1 import Model, weight_initial

from seq_encoding import ENCODING_METHOD

#############################################################################################
#
# Train
#
#############################################################################################


def batch_train(model, data):

    feature_seq, label = data
    pred_results = model(feature_seq)
    loss = nn.MSELoss()(pred_results, label)

    return loss


def batch_validation(model, data):
    with torch.no_grad():
        return batch_train(model, data)


def train(datapre,inv_size, v_size, input_channels, hidden_size, output_size, batch_size, layers):

    # log pytorch version
    log_to_file('PyTorch version', torch.__version__)

    # prepare model
    model = Model(inv_size, v_size, input_channels, hidden_size, output_size, batch_size, layers)
    weight_initial(model)
    # model.to(device)

    # OPTIMIZER
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    log_to_file("Optimizer", "SGD")

    # call backs
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.00001, patience=4,
                                               cooldown=4, verbose=True, min_lr=0.0001, factor=0.2)
    # some vars
    epoch_loss = 0
    validation_loss = 0
    steps = datapre.train_steps()

    log_to_file('Start training', datetime.datetime.now())
    for epoch in range(20):
        epoch_start_time = datetime.datetime.now()

        # train batches
        model.train(True)
        for _ in range(steps):
            data = datapre.batch_train_data()
            loss = batch_train(model, data)
            loss.backward()

            # clip grads
            nn.utils.clip_grad_value_(model.parameters(), 0.9)

            # update params
            optimizer.step()

            # record loss
            epoch_loss += loss.item()

            # reset grad
            optimizer.zero_grad()

        # time compute
        time_delta = datetime.datetime.now() - epoch_start_time

        # validation on epoch end
        model.eval()
        for _ in range(datapre.val_steps()):
            data = datapre.batch_val_data()
            validation_loss += batch_validation(model,
                                                data).item()

        # log
        log_to_file("Training process", "[Epoch {0:04d}] - time: {1:4d} s, train_loss: {2:0.5f}, val_loss: {3:0.5f}".format(
            epoch, time_delta.seconds, epoch_loss / steps, validation_loss / datapre.val_steps()))

        # LR schedule
        scheduler.step(loss.item())

        # reset loss
        epoch_loss = 0
        validation_loss = 0

        # reset data provider
        datapre.new_epoch()

    # save last epoch model
    torch.save(model.state_dict(), 'last_epoch_model.pytorch')

#############################################################################################
#
# Test
#
#############################################################################################


def test(datapre,inv_size, v_size, input_channels, hidden_size, output_size, batch_size, layers):
    """Test on weekly
    """
    # load and prepare model
    state_dict = torch.load('last_epoch_model.pytorch')
    model = Model(inv_size, v_size, input_channels, hidden_size, output_size, batch_size, layers)
    model.load_state_dict(state_dict)
    # model.to(device)
    model.eval()

    test_loss = 0
    for _ in range(datapre.test_steps()):
        data = datapre.batch_test_data()
        test_loss += batch_validation(model, data).item()

    
    log_to_file("test_loss",test_loss)





#############################################################################################
#
# Main
#
#############################################################################################


def main():
    
    # setup logger
    setup_logging()

    # encoding func
    feature_encoder = ENCODING_METHOD


    datapre = DataPreprocessing(
        feature_encoder,
        batch_size=10
    )
    log_to_file('Traning samples', len(datapre.train_samples))
    log_to_file('Val samples', len(datapre.validation_samples))
    log_to_file('Traning steps', datapre.train_steps())
    log_to_file('Val steps', datapre.val_steps())
    log_to_file('Batch size', datapre.batch_size)

    train(datapre,inv_size, v_size, input_channels, hidden_size, output_size, batch_size, layers)
    test( data_pre,datapre,inv_size, v_size, input_channels, hidden_size, output_size, batch_size, layers)



if __name__ == '__main__':
    main()