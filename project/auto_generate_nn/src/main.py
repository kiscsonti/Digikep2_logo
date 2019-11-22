import os
import time
import torch
import random
import numpy as np

from datetime import datetime

from doc import load_data
from config import args, copy_standard_config, setConstanst_Value
from model import Model

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


if __name__ == '__main__':

    #otthon
    # generated_data_path = "/media/kiscsonti/521493CD1493B289/egyetem/mester/1.felev/digikep/logo_class/Digikep2_logo/Generator/Linux/output/"
    # train_data_path = "/media/kiscsonti/521493CD1493B289/egyetem/mester/1.felev/digikep/logo_class/Digikep2_logo/Generator/Linux/train/"
    # test_data_path = "/media/kiscsonti/521493CD1493B289/egyetem/mester/1.felev/digikep/logo_class/Digikep2_logo/Generator/Linux/test/"


    #alagsor
    generated_data_path = "/home/petigep/college/orak/digikep2/Digikep2_logo/Generator/Linux/output"
    train_data_path = "/home/petigep/college/orak/digikep2/Digikep2_logo/Generator/Linux/train"
    test_data_path = "/home/petigep/college/orak/digikep2/Digikep2_logo/Generator/Linux/test"
    config_path = "/home/petigep/college/orak/digikep2/Digikep2_logo/Generator/Linux/config.json"

    epoch_counter = 0
    best_dev_acc = 0.0
    change_const_iter1 = 1000
    change_const_iter2 = 2000

    os.makedirs('./saved_model', exist_ok=True)
    best_model_path = './saved_model/best_model_%d.mdl' % (args.seed)
    best_modelweights_path = './saved_model/best_modelweights_%d.mdl' % (args.seed)

    last_model_path = './saved_model/lasttmp_model_%d.mdl' % (args.seed)
    last_modelweights_path = './saved_model/lasttmp_modelweights_%d.mdl' % (args.seed)

    print('Trained model will be saved to %s' % best_model_path)
    copy_standard_config(config_path)
    setConstanst_Value(0.75, config_path)

    test_data = load_data(generated_data_path, test_data_path, args=args, shuffle=False)

    labels_count = set()
    for item in test_data.dataset.classes:
        labels_count.add(item)
    print('Number of labels:', len(labels_count))
    model = Model(args)

    print('Epoch %d...' % epoch_counter)
    if epoch_counter == 0:
        dev_acc = model.evaluate(test_data)
        print('Dev accuracy: %f' % dev_acc)

    while True:

        if epoch_counter == change_const_iter1:
            setConstanst_Value(0.4, config_path)
        if epoch_counter == change_const_iter2:
            setConstanst_Value(0.05, config_path)

        train_data = load_data(generated_data_path, train_data_path, args=args)

        start_time = time.time()
        # np.random.shuffle(train_data)
        # cur_train_data = train_data
        for i in range(4):
            model.train(train_data)
            #ez kérdéses működni fog-e
            train_acc = model.evaluate(train_data, debug=False, eval_train=True, max_input=args.number_of_labels*args.batch_size)
            print('Train accuracy: %f' % train_acc)

        # if epoch_counter % args.check_per_epoch == 0:
        dev_acc = model.evaluate(test_data, debug=True)
        print('Dev accuracy: %f' % dev_acc)

        torch.save(model.network.state_dict(), last_modelweights_path)
        torch.save(model.network, last_model_path)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            # os.system('mv ./data/output.log ./data/best-dev.log')
            # model.save(best_model_path)
            torch.save(model.network.state_dict(), best_modelweights_path)
            # the_model = TheModelClass(*args, **kwargs)
            # the_model.load_state_dict(torch.load(PATH))
            torch.save(model.network, best_model_path)
                # the_model = torch.load(PATH)

        # test_data = load_data(generated_data_path, test_data_path, args=args, shuffle=False)
        # elif args.test_mode:
        #     model.save(checkpoint_path)
        print('Epoch %d use %d seconds.' % (epoch_counter, time.time() - start_time))
        epoch_counter += 1
        print('Best dev accuracy: %f' % best_dev_acc)
