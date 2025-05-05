import argparse
import json
import os

import numpy as np
from cgms_data_seg import CGMSDataSeg
from cnn_ohio import regressor, regressor_transfer, test_ckpt
from data_reader import DataReader


def personalized_train_ohio(epoch, ph, path="../output"):
    # read in all patients data
    pids = [559, 563, 570, 588, 575, 591]
    train_data = dict()
    test_data = []
    for pid in pids:
        reader = DataReader(
            "ohio", f"../data/{pid}-ws-training.xml", 5
        )
        train_data[pid] = reader.read()

    # add test data of 2018 patient
    standard = False  # do not use standard
    for pid in pids:
        reader = DataReader(
            "ohio", f"../data/{pid}-ws-testing.xml", 5
        )
        test_data += reader.read()

    # a dumb dataset instance
    train_dataset = CGMSDataSeg(
        "ohio", "../data/559-ws-training.xml", 5
    )
    sampling_horizon = 7
    prediction_horizon = ph
    scale = 0.01
    outtype = "Same"
    # train on training dataset
    # k_size, nblock, nn_size, nn_layer, learning_rate, batch_size, epoch, beta
    with open(os.path.join(path, "config.json")) as json_file:
        config = json.load(json_file)
    argv = (
        config["k_size"],
        config["nblock"],
        config["nn_size"],
        config["nn_layer"],
        config["learning_rate"],
        config["batch_size"],
        epoch,
        config["beta"],
    )
    l_type = config["loss"]
    print(l_type)
    # test on patients data
    outdir = os.path.join(path, f"ph_{prediction_horizon}_{l_type}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    all_errs = []
    for pid in pids:
        # 100 is dumb if set_cutpoint is used
        train_pids = pids
        local_train_data = test_data

        # oversample the training data
        for k in train_pids:
            local_train_data += train_data[k]

        print(f"Pretrain data: {sum([sum(x) for x in local_train_data])}")
        train_dataset.data = local_train_data
        train_dataset.set_cutpoint = -1
        train_dataset.reset(
            sampling_horizon,
            prediction_horizon,
            scale,
            100,
            False,
            outtype,
            1,
            standard,
        )
        regressor(train_dataset, *argv, l_type, outdir)
        # fine-tune on personal data
        target_test_dataset = CGMSDataSeg(
            "ohio", f"../data/{pid}-ws-testing.xml", 5
        )
        target_test_dataset.set_cutpoint = 1
        target_test_dataset.reset(
            sampling_horizon,
            prediction_horizon,
            scale,
            0.01,
            False,
            outtype,
            1,
            standard,
        )
        target_train_dataset = CGMSDataSeg(
            "ohio", f"../data/{pid}-ws-training.xml", 5
        )

        target_train_dataset.set_cutpoint = -1
        target_train_dataset.reset(
            sampling_horizon,
            prediction_horizon,
            scale,
            100,
            False,
            outtype,
            1,
            standard,
        )
        err, labels = test_ckpt(target_test_dataset, outdir)
        errs = [err]
        transfer_res = [labels]
        for i in range(1, 4):
            err, labels = regressor_transfer(
                target_train_dataset,
                target_test_dataset,
                config["batch_size"],
                epoch,
                outdir,
                i,
            )
            errs.append(err)
            transfer_res.append(labels)
        transfer_res = np.concatenate(transfer_res, axis=1)
        np.savetxt(
            f"{outdir}/{pid}.txt",
            transfer_res,
            fmt="%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
        )
        all_errs.append([pid] + errs)
    all_errs = np.array(all_errs)
    np.savetxt(f"{outdir}/errors.txt", all_errs, fmt="%d %.4f %.4f %.4f %.4f")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--prediction_horizon", type=int, default=6)
    parser.add_argument("--outdir", type=str, default="../ohio_results")
    args = parser.parse_args()

    personalized_train_ohio(args.epoch, args.prediction_horizon, args.outdir)


if __name__ == "__main__":
    main()
