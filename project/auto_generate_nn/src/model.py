import logging
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from models import BasicNet_14, LinearNet, CNN_32_64_32

logger = logging.getLogger()


class Model:

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.use_cuda = (args.use_cuda == True) and torch.cuda.is_available()
        print('Use cuda:', self.use_cuda)
        if self.use_cuda:
            torch.cuda.set_device(int(args.gpu))
        self.network = CNN_32_64_32(args)
        self.init_optimizer()
        if args.pretrained:
            print('Load pretrained model from %s...' % args.pretrained)
            self.load(args.pretrained)
        if self.use_cuda:
            self.network.to('cuda')
            # self.network.cuda()
        print(self.network)
        self._report_num_trainable_parameters()

    def _report_num_trainable_parameters(self):
        num_parameters = 0
        for p in self.network.parameters():
            if p.requires_grad:
                sz = list(p.size())
                num_parameters += np.prod(sz)
        print('Number of parameters: ', num_parameters)

    def train(self, train_data):
        self.network.train()
        self.updates = 0
        iter_cnt, num_iter = 0, len(train_data)
        for inputs, labels, _ in train_data:

            if self.use_cuda:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

            # print(inputs)
            # print("Labels: ", labels)
            # print("_:", _)
            # import sys
            # sys.exit(0)
            pred_proba = self.network(inputs)
            loss = F.nll_loss(pred_proba, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.grad_clipping)

            # Update parameters
            self.optimizer.step()
            self.updates += 1
            iter_cnt += 1

            if self.updates % 20 == 0:
                print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter, loss.item()))
        self.scheduler.step()
        print('LR:', self.scheduler.get_lr()[0])

    def evaluate(self, dev_data, debug=False, eval_train=False, max_input=None):
        self.network.eval()
        predictions = []
        actuals = []
        all_img_names = []
        good = 0
        total = 0
        nmb_of_input = 0
        for inputs, labels, image_names in dev_data:
            if max_input is not None and nmb_of_input >= max_input:
                break

            nmb_of_input += len(image_names)
            if self.use_cuda:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
            pred_proba = self.network(inputs)

            actuals.extend(labels)
            all_img_names.extend(image_names)

            final_predicts = torch.argmax(pred_proba, dim=1)
            predictions.extend(final_predicts)
            total += len(labels)
            for a, b in zip(final_predicts, labels):
                if a == b:
                    good += 1

        if eval_train:
            return good/total

        if debug:
            writer = open('../data/output.log', 'w', encoding='utf-8')
            writer.write('Image name, Actual label, Predicted label\r\n')
            for i in range(len(all_img_names)):
                writer.write('%s, %s, %s\r\n'.format(all_img_names[i], str(actuals[i]), str(predictions[i])))
        acc = 1.0 * good / total
        if debug:
            writer.write('Accuracy: %f\n' % acc)
            writer.close()
        return acc

    # def predict(self, test_data):
    #     # DO NOT SHUFFLE test_data
    #     self.network.eval()
    #     prediction = []
    #     for batch_input in self._iter_data(test_data):
    #         feed_input = [x for x in batch_input[:-1]]
    #         pred_proba = self.network(*feed_input)
    #         pred_proba = pred_proba.data.cpu()
    #         prediction += list(pred_proba)
    #     return prediction

    # def _iter_data(self, data):
    #     num_iter = (len(data) + self.batch_size - 1) // self.batch_size
    #     for i in range(num_iter):
    #         start_idx = i * self.batch_size
    #         batch_data = data[start_idx:(start_idx + self.batch_size)]
    #         batch_input = batchify(batch_data)
    #
    #         # Transfer to GPU
    #         if self.use_cuda:
    #             batch_input = [Variable(x.cuda(async=True)) for x in batch_input]
    #         else:
    #             batch_input = [Variable(x) for x in batch_input]
    #         yield batch_input

    def init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.lr,
                                       momentum=0.4,
                                       weight_decay=0)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                        lr=self.lr,
                                        weight_decay=0)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 80, 300], gamma=0.5)

    def save(self, ckt_path):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {'state_dict': state_dict}
        torch.save(params, ckt_path)

    def load(self, ckt_path):
        logger.info('Loading model %s' % ckt_path)
        saved_params = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        return self.network.load_state_dict(state_dict)

    def cuda(self):
        self.use_cuda = True
        self.network.cuda()
