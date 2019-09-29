import os
import time
import torch
import random
import numpy as np

from datetime import datetime

from doc import load_data
from config import args
from model import Model

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if __name__ == '__main__':

    train_data_path = '../../data/train_data_merged.csv'
    test_data_path = '../../data/test_data_merged.csv'
    img_folder = '../../data/flickr_logos_27_dataset/flickr_logos_27_dataset_images'
    train_data = load_data(train_data_path, img_folder, args)
    test_data = load_data(test_data_path, img_folder, args=args, shuffle=False)

    labels_count = set()
    for item in train_data.dataset.y:
        labels_count.add(item)
    print('Number of labels:', len(labels_count))

    model = Model(args, len(labels_count))

    best_dev_acc = 0.0
    os.makedirs('./checkpoint', exist_ok=True)
    checkpoint_path = './checkpoint/%d-%s.mdl' % (args.seed, datetime.now().isoformat())
    print('Trained model will be saved to %s' % checkpoint_path)

    for i in range(args.epoch):
        print('Epoch %d...' % i)
        if i == 0:
            dev_acc = model.evaluate(test_data)
            print('Dev accuracy: %f' % dev_acc)
        start_time = time.time()
        # np.random.shuffle(train_data)
        # cur_train_data = train_data

        model.train(train_data)
        #ez kérdéses működni fog-e
        train_acc = model.evaluate(train_data, debug=False, eval_train=True, max_input=250)
        print('Train accuracy: %f' % train_acc)
        dev_acc = model.evaluate(test_data, debug=True)
        print('Dev accuracy: %f' % dev_acc)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.system('mv ./data/output.log ./data/best-dev.log')
            model.save(checkpoint_path)
        # elif args.test_mode:
        #     model.save(checkpoint_path)
        print('Epoch %d use %d seconds.' % (i, time.time() - start_time))

    print('Best dev accuracy: %f' % best_dev_acc)
