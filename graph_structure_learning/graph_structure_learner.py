"""
    Performs graph structure learning on the MSG as outlined in the
    paper.
"""
import os
import time
import json
import glob
import numpy as np
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from collections import Counter
from sklearn.metrics import r2_score
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


# from .model import Model
# from .utils.generic_utils import to_cuda
# from .utils.data_utils import prepare_datasets, DataStream, vectorize_input
# from .utils import Timer, DummyLogger, AverageMeter
# from .utils import constants as Constants
# from .layers.common import dropout
# from .layers.anchor import sample_anchors, batch_sample_anchors, batch_select_from_tensor,

# from .models.graph_clf import GraphClf
# from .models.text_graph import TextGraphRegression, TextGraphClf
# from .utils.text_data.vocab_utils import VocabModel
# from .utils import constants as Constants
# from .utils.generic_utils import to_cuda, create_mask
# from .utils.constants import INF
# from .utils.radam import RAdam

# from .model import Model
# from .utils.generic_utils import to_cuda
# from .utils.data_utils import prepare_datasets, DataStream, vectorize_input
# from .utils import Timer, DummyLogger, AverageMeter
# from .utils import constants as Constants
# from .layers.common import dropout
# from .layers.anchor import sample_anchors, batch_sample_anchors, batch_select_from_tensor, compute_anchor_adj

VERY_SMALL_NUMBER = 1e-12
INF = 1e20


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


class GraphLearner(nn.Module):
    """
        Implements GSL.
    """

    def __init__(self):
        def __init__(self, input_size, hidden_size, topk=None, epsilon=None, num_pers=16, metric_type='weighted_cosine', device=None):
            super(GraphLearner, self).__init__()
            self.device = device
            self.topk = topk
            self.epsilon = epsilon
            self.metric_type = metric_type

            # if metric_type == 'weighted_cosine'
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(
                nn.init.xavier_uniform_(self.weight_tensor))
            print(
                '[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

    def forward(self, context, ctx_mask=None):
        # if metric_type == 'weighted_cosine'
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        if len(context.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

        context_fc = context.unsqueeze(0) * expand_weight_tensor
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        attention = torch.matmul(
            context_norm, context_norm.transpose(-1, -2)).mean(0)
        markoff_value = 0

        # if ctx_mask is not None:
        #     attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
        #     attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(
                attention, self.epsilon, markoff_value)

        # if self.topk is not None:
        #     attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)

        return attention

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * \
            mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix


class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set=None):
        self.config = config
        if self.config['model_name'] == 'GraphClf':
            self.net_module = GraphClf
        elif self.config['model_name'] == 'TextGraphRegression':
            self.net_module = TextGraphRegression
        elif self.config['model_name'] == 'TextGraphClf':
            self.net_module = TextGraphClf
        else:
            raise RuntimeError('Unknown model_name: {}'.format(
                self.config['model_name']))
        print('[ Running {} model ]'.format(self.config['model_name']))

        if config['data_type'] == 'text':
            saved_vocab_file = os.path.join(config['data_dir'], '{}_seed{}.vocab'.format(
                config['dataset_name'], config.get('data_seed', 1234)))
            self.vocab_model = VocabModel.build(
                saved_vocab_file, train_set, self.config)

        if config['task_type'] == 'regression':
            assert config['out_predictions']
            self.criterion = F.mse_loss
            self.score_func = r2_score
            self.metric_name = 'r2'
        elif config['task_type'] == 'classification':
            self.criterion = F.nll_loss
            self.score_func = accuracy
            self.metric_name = 'acc'
        else:
            self.criterion = F.nll_loss
            self.score_func = None
            self.metric_name = None

        if self.config['pretrained']:
            self.init_saved_network(self.config['pretrained'])
        else:
            # Building network.
            self._init_new_network()

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()

        print('#Parameters = {}\n'.format(num_params))
        self._init_optimizer()

    def init_saved_network(self, saved_dir):
        _ARGUMENTS = ['word_embed_dim', 'hidden_size', 'f_qem', 'f_pos', 'f_ner',
                      'word_dropout', 'rnn_dropout',
                      'ctx_graph_hops', 'ctx_graph_topk',
                      'score_unk_threshold', 'score_yes_threshold',
                      'score_no_threshold']

        # Load all saved fields.
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(
            fname, map_location=lambda storage, loc: storage)
        self.state_dict = saved_params['state_dict']
        # for k in _ARGUMENTS:
        #     if saved_params['config'][k] != self.config[k]:
        #         print('Overwrite {}: {} -> {}'.format(k, self.config[k], saved_params['config'][k]))
        #         self.config[k] = saved_params['config'][k]

        if self.config['data_type'] == 'text':
            w_embedding = self._init_embedding(
                len(self.vocab_model.word_vocab), self.config['word_embed_dim'])
            self.network = self.net_module(
                self.config, w_embedding, self.vocab_model.word_vocab)
        else:
            self.network = self.net_module(self.config)

        # Merge the arguments
        if self.state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

    def _init_new_network(self):
        if self.config['data_type'] == 'text':
            w_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'],
                                               pretrained_vecs=self.vocab_model.word_vocab.embeddings)
            self.network = self.net_module(
                self.config, w_embedding, self.vocab_model.word_vocab)
        else:
            self.network = self.net_module(self.config)

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(
                parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'radam':
            self.optimizer = RAdam(
                parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.config['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_reduce_factor'],
                                           patience=self.config['lr_patience'], verbose=True)

    def _init_embedding(self, vocab_size, embed_size, pretrained_vecs=None):
        """Initializes the embeddings
        """
        return nn.Embedding(vocab_size, embed_size, padding_idx=0,
                            _weight=torch.from_numpy(pretrained_vecs).float()
                            if pretrained_vecs is not None else None)

    def save(self, dirname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'config': self.config,
            'dir': dirname,
        }
        try:
            torch.save(params, os.path.join(
                dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')

    def clip_grad(self):
        # Clip gradients
        if self.config['grad_clipping']:
            parameters = [p for p in self.network.parameters()
                          if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(
                parameters, self.config['grad_clipping'])


def train_batch(batch, network, vocab, criterion, forcing_ratio, rl_ratio, config, wmd=None):
    network.train(True)

    with torch.set_grad_enabled(True):
        ext_vocab_size = batch['oov_dict'].ext_vocab_size if batch['oov_dict'] else None

        network_out = network(batch, batch['targets'], criterion,
                              forcing_ratio=forcing_ratio, partial_forcing=config['partial_forcing'],
                              sample=config['sample'], ext_vocab_size=ext_vocab_size,
                              include_cover_loss=config['show_cover_loss'])

        if rl_ratio > 0:
            batch_size = batch['context'].shape[0]
            sample_out = network(batch, saved_out=network_out, criterion=criterion,
                                 criterion_reduction=False, criterion_nll_only=True,
                                 sample=True, ext_vocab_size=ext_vocab_size)
            baseline_out = network(batch, saved_out=network_out, visualize=False,
                                   ext_vocab_size=ext_vocab_size)

            sample_out_decoded = sample_out.decoded_tokens.transpose(0, 1)
            baseline_out_decoded = baseline_out.decoded_tokens.transpose(0, 1)

            neg_reward = []
            for i in range(batch_size):
                scores = eval_batch_output([batch['target_src'][i]], vocab, batch['oov_dict'],
                                           [sample_out_decoded[i]], [baseline_out_decoded[i]])

                greedy_score = scores[1][config['rl_reward_metric']]
                reward_ = scores[0][config['rl_reward_metric']] - greedy_score

                if config['rl_wmd_ratio'] > 0:
                    # Add word mover's distance
                    sample_seq = batch_decoded_index2word(
                        [sample_out_decoded[i]], vocab, batch['oov_dict'])[0]
                    greedy_seq = batch_decoded_index2word(
                        [baseline_out_decoded[i]], vocab, batch['oov_dict'])[0]

                    sample_wmd = - \
                        wmd.distance(
                            sample_seq, batch['target_src'][i]) / max(len(sample_seq.split()), 1)
                    greedy_wmd = - \
                        wmd.distance(
                            greedy_seq, batch['target_src'][i]) / max(len(greedy_seq.split()), 1)
                    wmd_reward_ = sample_wmd - greedy_wmd
                    wmd_reward_ = max(
                        min(wmd_reward_, config['max_wmd_reward']), -config['max_wmd_reward'])
                    reward_ += config['rl_wmd_ratio'] * wmd_reward_

                neg_reward.append(reward_)
            neg_reward = to_cuda(torch.Tensor(neg_reward), network.device)

            # if sample > baseline, the reward is positive (i.e. good exploration), rl_loss is negative
            rl_loss = torch.sum(neg_reward * sample_out.loss) / batch_size
            rl_loss_value = torch.sum(
                neg_reward * sample_out.loss_value).item() / batch_size
            loss = (1 - rl_ratio) * network_out.loss + rl_ratio * rl_loss
            loss_value = (1 - rl_ratio) * network_out.loss_value + \
                rl_ratio * rl_loss_value

            metrics = eval_batch_output(batch['target_src'], vocab,
                                        batch['oov_dict'], baseline_out.decoded_tokens)[0]

        else:
            loss = network_out.loss
            loss_value = network_out.loss_value
            metrics = eval_batch_output(batch['target_src'], vocab,
                                        batch['oov_dict'], network_out.decoded_tokens)[0]

    return loss, loss_value, metrics


def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return correct / len(labels)


class ModelHandler(object):
    """High level model_handler that trains/validates/tests the network,
    tracks and logs metrics.
    """

    def __init__(self, config):
        # Evaluation Metrics:
        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()
        if config['task_type'] == 'classification':
            self._train_metrics = {'nloss': AverageMeter(),
                                   'acc': AverageMeter()}
            self._dev_metrics = {'nloss': AverageMeter(),
                                 'acc': AverageMeter()}
        elif config['task_type'] == 'regression':
            self._train_metrics = {'nloss': AverageMeter(),
                                   'r2': AverageMeter()}
            self._dev_metrics = {'nloss': AverageMeter(),
                                 'r2': AverageMeter()}
        else:
            raise ValueError(
                'Unknown task_type: {}'.format(config['task_type']))

        self.logger = DummyLogger(
            config, dirname=config['out_dir'], pretrained=config['pretrained'])
        self.dirname = self.logger.dirname
        if not config['no_cuda'] and torch.cuda.is_available():
            print('[ Using CUDA ]')
            self.device = torch.device(
                'cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        config['device'] = self.device

        seed = config.get('seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device:
            torch.cuda.manual_seed(seed)

        datasets = prepare_datasets(config)

        # Prepare datasets
        if config['data_type'] in ('network', 'uci'):
            config['num_feat'] = datasets['features'].shape[-1]
            config['num_class'] = datasets['labels'].max().item() + 1

            # Initialize the model
            self.model = Model(config, train_set=datasets.get('train', None))
            self.model.network = self.model.network.to(self.device)

            self._n_test_examples = datasets['idx_test'].shape[0]
            self.run_epoch = self._scalable_run_whole_epoch if config.get(
                'scalable_run', False) else self._run_whole_epoch

            self.train_loader = datasets
            self.dev_loader = datasets
            self.test_loader = datasets

        else:
            train_set = datasets['train']
            dev_set = datasets['dev']
            test_set = datasets['test']

            config['num_class'] = max(
                [x[-1] for x in train_set + dev_set + test_set]) + 1

            self.run_epoch = self._run_batch_epoch

            # Initialize the model
            self.model = Model(config, train_set=datasets.get('train', None))
            self.model.network = self.model.network.to(self.device)

            self._n_train_examples = 0
            if train_set:
                self.train_loader = DataStream(
                    train_set, self.model.vocab_model.word_vocab, config=config, isShuffle=True, isLoop=True, isSort=True)
                self._n_train_batches = self.train_loader.get_num_batch()
            else:
                self.train_loader = None

            if dev_set:
                self.dev_loader = DataStream(
                    dev_set, self.model.vocab_model.word_vocab, config=config, isShuffle=False, isLoop=True, isSort=True)
                self._n_dev_batches = self.dev_loader.get_num_batch()
            else:
                self.dev_loader = None

            if test_set:
                self.test_loader = DataStream(test_set, self.model.vocab_model.word_vocab, config=config,
                                              isShuffle=False, isLoop=False, isSort=True, batch_size=config['batch_size'])
                self._n_test_batches = self.test_loader.get_num_batch()
                self._n_test_examples = len(test_set)
            else:
                self.test_loader = None

        self.config = self.model.config
        self.is_test = False

    def train(self):
        if self.train_loader is None or self.dev_loader is None:
            print("No training set or dev set specified -- skipped training.")
            return

        self.is_test = False
        timer = Timer("Train")
        self._epoch = self._best_epoch = 0

        self._best_metrics = {}
        for k in self._dev_metrics:
            self._best_metrics[k] = -float('inf')
        self._reset_metrics()

        while self._stop_condition(self._epoch, self.config['patience']):
            self._epoch += 1

            # Train phase
            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "\n>>> Train Epoch: [{} / {}]".format(
                    self._epoch, self.config['max_epochs'])
                print(format_str)
                self.logger.write_to_file(format_str)

            self.run_epoch(self.train_loader, training=True,
                           verbose=self.config['verbose'])
            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "Training Epoch {} -- Loss: {:0.5f}".format(
                    self._epoch, self._train_loss.mean())
                format_str += self.metric_to_str(self._train_metrics)
                train_epoch_time_msg = timer.interval(
                    "Training Epoch {}".format(self._epoch))
                self.logger.write_to_file(
                    train_epoch_time_msg + '\n' + format_str)
                print(format_str)
                format_str = "\n>>> Validation Epoch: [{} / {}]".format(
                    self._epoch, self.config['max_epochs'])
                print(format_str)
                self.logger.write_to_file(format_str)

            # Validation phase
            dev_output, dev_gold = self.run_epoch(self.dev_loader, training=False, verbose=self.config['verbose'],
                                                  out_predictions=self.config['out_predictions'])
            if self.config['out_predictions']:
                dev_metric_score = self.model.score_func(dev_gold, dev_output)
            else:
                dev_metric_score = None

            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "Validation Epoch {} -- Loss: {:0.5f}".format(
                    self._epoch, self._dev_loss.mean())
                format_str += self.metric_to_str(self._dev_metrics)
                if dev_metric_score is not None:
                    format_str += '\n Dev score: {:0.5f}'.format(
                        dev_metric_score)
                dev_epoch_time_msg = timer.interval(
                    "Validation Epoch {}".format(self._epoch))
                self.logger.write_to_file(
                    dev_epoch_time_msg + '\n' + format_str)
                print(format_str)

            if not self.config['data_type'] in ('network', 'uci', 'text'):
                self.model.scheduler.step(
                    self._dev_metrics[self.config['eary_stop_metric']].mean())

            if self.config['eary_stop_metric'] == self.model.metric_name and dev_metric_score is not None:
                cur_dev_score = dev_metric_score
            else:
                cur_dev_score = self._dev_metrics[self.config['eary_stop_metric']].mean(
                )

            # if self._best_metrics[self.config['eary_stop_metric']] < self._dev_metrics[self.config['eary_stop_metric']].mean():
            if self._best_metrics[self.config['eary_stop_metric']] < cur_dev_score:
                self._best_epoch = self._epoch
                for k in self._dev_metrics:
                    self._best_metrics[k] = self._dev_metrics[k].mean()

                if dev_metric_score is not None:
                    self._best_metrics[self.model.metric_name] = dev_metric_score

                if self.config['save_params']:
                    self.model.save(self.dirname)
                    if self._epoch % self.config['print_every_epochs'] == 0:
                        format_str = 'Saved model to {}'.format(self.dirname)
                        self.logger.write_to_file(format_str)
                        print(format_str)

                if self._epoch % self.config['print_every_epochs'] == 0:
                    format_str = "!!! Updated: " + \
                        self.best_metric_to_str(self._best_metrics)
                    self.logger.write_to_file(format_str)
                    print(format_str)

            self._reset_metrics()

        timer.finish()

        format_str = "Finished Training: {}\nTraining time: {}".format(
            self.dirname, timer.total) + '\n' + self.summary()
        print(format_str)
        self.logger.write_to_file(format_str)
        return self._best_metrics

    def test(self):
        if self.test_loader is None:
            print("No testing set specified -- skipped testing.")
            return

        # Restore best model
        print('Restoring best model')
        self.model.init_saved_network(self.dirname)
        self.model.network = self.model.network.to(self.device)

        self.is_test = True
        self._reset_metrics()
        timer = Timer("Test")
        for param in self.model.network.parameters():
            param.requires_grad = False

        output, gold = self.run_epoch(self.test_loader, training=False, verbose=0,
                                      out_predictions=self.config['out_predictions'])

        metrics = self._dev_metrics
        format_str = "[test] | test_exs = {} | step: [{} / {}]".format(
            self._n_test_examples, 1, 1)
        format_str += self.metric_to_str(metrics)

        if self.config['out_predictions']:
            test_score = self.model.score_func(gold, output)
            format_str += '\nFinal score on the testing set: {:0.5f}\n'.format(
                test_score)
        else:
            test_score = None

        print(format_str)
        self.logger.write_to_file(format_str)
        timer.finish()

        format_str = "Finished Testing: {}\nTesting time: {}".format(
            self.dirname, timer.total)
        print(format_str)
        self.logger.write_to_file(format_str)
        self.logger.close()

        test_metrics = {}
        for k in metrics:
            test_metrics[k] = metrics[k].mean()

        if test_score is not None:
            test_metrics[self.model.metric_name] = test_score
        return test_metrics

    def batch_no_gnn(self, x_batch, step, training, out_predictions=False):
        '''Iterative graph learning: batch training'''
        mode = "train" if training else ("test" if self.is_test else "dev")
        network = self.model.network
        network.train(training)

        context, context_lens, targets = x_batch['context'], x_batch['context_lens'], x_batch['targets']
        context2 = x_batch.get('context2', None)
        context2_lens = x_batch.get('context2_lens', None)

        output = network.compute_no_gnn_output(context, context_lens)

        # BP to update weights
        loss = self.model.criterion(output, targets)
        score = self.model.score_func(targets.cpu(), output.detach().cpu())

        res = {'loss': loss.item(),
               'metrics': {'nloss': -loss.item(), self.model.metric_name: score},
               }
        if out_predictions:
            res['predictions'] = output.detach().cpu()

        if training:
            # Normalize our loss (if averaged)
            loss = loss / self.config['grad_accumulated_steps']
            loss.backward()

            # Wait for several backward steps
            if (step + 1) % self.config['grad_accumulated_steps'] == 0:
                self.model.clip_grad()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
        return res

    def batch_IGL_stop(self, x_batch, step, training, out_predictions=False):
        '''Iterative graph learning: batch training, batch stopping'''
        mode = "train" if training else ("test" if self.is_test else "dev")
        network = self.model.network
        network.train(training)

        context, context_lens, targets = x_batch['context'], x_batch['context_lens'], x_batch['targets']
        context2 = x_batch.get('context2', None)
        context2_lens = x_batch.get('context2_lens', None)

        # Prepare init node embedding, init adj
        raw_context_vec, context_vec, context_mask, init_adj = network.prepare_init_graph(
            context, context_lens)

        # Init
        raw_node_vec = raw_context_vec  # word embedding
        init_node_vec = context_vec  # hidden embedding
        node_mask = context_mask

        cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner, raw_node_vec, network.graph_skip_conn,
                                                   node_mask=node_mask, graph_include_self=network.graph_include_self, init_adj=init_adj)
        node_vec = torch.relu(
            network.encoder.graph_encoders[0](init_node_vec, cur_adj))
        node_vec = F.dropout(node_vec, network.dropout,
                             training=network.training)

        # Add mid GNN layers
        for encoder in network.encoder.graph_encoders[1:-1]:
            node_vec = torch.relu(encoder(node_vec, cur_adj))
            node_vec = F.dropout(node_vec, network.dropout,
                                 training=network.training)

        # BP to update weights
        output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
        output = network.compute_output(output, node_mask=node_mask)
        loss1 = self.model.criterion(output, targets)
        score = self.model.score_func(targets.cpu(), output.detach().cpu())

        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            loss1 += self.add_batch_graph_loss(cur_raw_adj, raw_node_vec)

        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        if not mode == 'test':
            if self._epoch > self.config.get('pretrain_epoch', 0):
                max_iter_ = self.config.get('max_iter', 10)  # Fine-tuning
                if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')

            else:
                max_iter_ = 0  # Pretraining
        else:
            max_iter_ = self.config.get('max_iter', 10)

        eps_adj = float(self.config.get('eps_adj', 0)) if training else float(
            self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))
        pre_raw_adj = cur_raw_adj
        pre_adj = cur_adj

        loss = 0
        iter_ = 0
        # Indicate the last iteration number for each example
        batch_last_iters = to_cuda(torch.zeros(
            x_batch['batch_size'], dtype=torch.uint8), self.device)
        # Indicate either an example is in onging state (i.e., 1) or stopping state (i.e., 0)
        batch_stop_indicators = to_cuda(torch.ones(
            x_batch['batch_size'], dtype=torch.uint8), self.device)
        batch_all_outputs = []
        while self.config['graph_learn'] and (iter_ == 0 or torch.sum(batch_stop_indicators).item() > 0) and iter_ < max_iter_:
            iter_ += 1
            batch_last_iters += batch_stop_indicators
            pre_adj = cur_adj
            pre_raw_adj = cur_raw_adj
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner2, node_vec, network.graph_skip_conn,
                                                       node_mask=node_mask, graph_include_self=network.graph_include_self, init_adj=init_adj)

            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_adj = update_adj_ratio * cur_adj + \
                    (1 - update_adj_ratio) * first_adj

            node_vec = torch.relu(
                network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, self.config.get(
                'gl_dropout', 0), training=network.training)

            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, self.config.get(
                    'gl_dropout', 0), training=network.training)

            # BP to update weights
            tmp_output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
            tmp_output = network.compute_output(
                tmp_output, node_mask=node_mask)
            batch_all_outputs.append(tmp_output.unsqueeze(1))

            tmp_loss = self.model.criterion(
                tmp_output, targets, reduction='none')
            if len(tmp_loss.shape) == 2:
                tmp_loss = torch.mean(tmp_loss, 1)

            loss += batch_stop_indicators.float() * tmp_loss

            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                loss += batch_stop_indicators.float() * self.add_batch_graph_loss(cur_raw_adj,
                                                                                  raw_node_vec, keep_batch_dim=True)

            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += batch_stop_indicators.float() * batch_SquaredFrobeniusNorm(cur_raw_adj -
                                                                                   pre_raw_adj) * self.config.get('graph_learn_ratio')

            tmp_stop_criteria = batch_diff(
                cur_raw_adj, pre_raw_adj, first_raw_adj) > eps_adj
            batch_stop_indicators = batch_stop_indicators * tmp_stop_criteria

        if iter_ > 0:
            loss = torch.mean(loss / batch_last_iters.float()) + loss1

            batch_all_outputs = torch.cat(batch_all_outputs, 1)
            selected_iter_index = batch_last_iters.long().unsqueeze(-1) - 1

            if len(batch_all_outputs.shape) == 3:
                selected_iter_index = selected_iter_index.unsqueeze(
                    -1).expand(-1, -1, batch_all_outputs.size(-1))
                output = batch_all_outputs.gather(
                    1, selected_iter_index).squeeze(1)
            else:
                output = batch_all_outputs.gather(1, selected_iter_index)

            score = self.model.score_func(targets.cpu(), output.detach().cpu())

        else:
            loss = loss1

        res = {'loss': loss.item(),
               'metrics': {'nloss': -loss.item(), self.model.metric_name: score},
               }
        if out_predictions:
            res['predictions'] = output.detach().cpu()

        if training:
            # Normalize our loss (if averaged)
            loss = loss / self.config['grad_accumulated_steps']
            loss.backward()

            # Wait for several backward steps
            if (step + 1) % self.config['grad_accumulated_steps'] == 0:
                self.model.clip_grad()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
        return res

    # def scalable_batch_IGL_stop(self, x_batch, step, training, out_predictions=False):
    #     '''Iterative graph learning: batch training, batch stopping'''
    #     mode = "train" if training else ("test" if self.is_test else "dev")
    #     network = self.model.network
    #     network.train(training)

    #     context, context_lens, targets = x_batch['context'], x_batch['context_lens'], x_batch['targets']
    #     context2 = x_batch.get('context2', None)
    #     context2_lens = x_batch.get('context2_lens', None)

    #     # Prepare init node embedding, init adj
    #     raw_context_vec, context_vec, context_mask, init_adj = network.prepare_init_graph(
    #         context, context_lens)

    #     # Init
    #     raw_node_vec = raw_context_vec  # word embedding
    #     init_node_vec = context_vec  # hidden embedding
    #     node_mask = context_mask

    #     # Randomly sample s anchor nodes
    #     init_anchor_vec, anchor_mask, sampled_node_idx, max_num_anchors = batch_sample_anchors(
    #         init_node_vec, network.config.get('ratio_anchors', 0.2), node_mask=node_mask, device=self.device)
    #     raw_anchor_vec = batch_select_from_tensor(
    #         raw_node_vec, sampled_node_idx, max_num_anchors, self.device)

    #     # Compute n x s node-anchor relationship matrix
    #     cur_node_anchor_adj = network.learn_graph(
    #         network.graph_learner, raw_node_vec, anchor_features=raw_anchor_vec, node_mask=node_mask, anchor_mask=anchor_mask)

    #     # Compute s x s anchor graph
    #     cur_anchor_adj = compute_anchor_adj(
    #         cur_node_anchor_adj, anchor_mask=anchor_mask)

    #     # Update node embeddings via node-anchor-node message passing
    #     init_agg_vec = network.encoder.graph_encoders[0](
    #         init_node_vec, init_adj, anchor_mp=False, batch_norm=False)
    #     node_vec = (1 - network.graph_skip_conn) * network.encoder.graph_encoders[0](init_node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
    #         network.graph_skip_conn * init_agg_vec

    #     if network.encoder.graph_encoders[0].bn is not None:
    #         node_vec = network.encoder.graph_encoders[0].compute_bn(node_vec)

    #     node_vec = torch.relu(node_vec)
    #     node_vec = F.dropout(node_vec, network.dropout,
    #                          training=network.training)
    #     anchor_vec = batch_select_from_tensor(
    #         node_vec, sampled_node_idx, max_num_anchors, self.device)

    #     first_node_anchor_adj, first_anchor_adj = cur_node_anchor_adj, cur_anchor_adj
    #     first_init_agg_vec = network.encoder.graph_encoders[0](
    #         init_node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)

    #     # Add mid GNN layers
    #     for encoder in network.encoder.graph_encoders[1:-1]:
    #         node_vec = (1 - network.graph_skip_conn) * encoder(node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
    #             network.graph_skip_conn * \
    #             encoder(node_vec, init_adj, anchor_mp=False, batch_norm=False)

    #         if encoder.bn is not None:
    #             node_vec = encoder.compute_bn(node_vec)

    #         node_vec = torch.relu(node_vec)
    #         node_vec = F.dropout(node_vec, network.dropout,
    #                              training=network.training)
    #         anchor_vec = batch_select_from_tensor(
    #             node_vec, sampled_node_idx, max_num_anchors, self.device)

    #     # Compute output via node-anchor-node message passing
    #     output = (1 - network.graph_skip_conn) * network.encoder.graph_encoders[-1](node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
    #         network.graph_skip_conn * network.encoder.graph_encoders[-1](
    #             node_vec, init_adj, anchor_mp=False, batch_norm=False)
    #     output = network.compute_output(output, node_mask=node_mask)
    #     loss1 = self.model.criterion(output, targets)
    #     score = self.model.score_func(targets.cpu(), output.detach().cpu())

    #     if self.config['graph_learn'] and self.config['graph_learn_regularization']:
    #         loss1 += self.add_batch_graph_loss(cur_anchor_adj, raw_anchor_vec)

    #     if not mode == 'test':
    #         if self._epoch > self.config.get('pretrain_epoch', 0):
    #             max_iter_ = self.config.get('max_iter', 10)  # Fine-tuning
    #             if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
    #                 for k in self._dev_metrics:
    #                     self._best_metrics[k] = -float('inf')

    #         else:
    #             max_iter_ = 0  # Pretraining
    #     else:
    #         max_iter_ = self.config.get('max_iter', 10)

    #     eps_adj = float(self.config.get('eps_adj', 0)) if training else float(
    #         self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))
    #     pre_node_anchor_adj = cur_node_anchor_adj

    #     loss = 0
    #     iter_ = 0
    #     # Indicate the last iteration number for each example
    #     batch_last_iters = to_cuda(torch.zeros(
    #         x_batch['batch_size'], dtype=torch.uint8), self.device)
    #     # Indicate either an example is in onging state (i.e., 1) or stopping state (i.e., 0)
    #     batch_stop_indicators = to_cuda(torch.ones(
    #         x_batch['batch_size'], dtype=torch.uint8), self.device)
    #     batch_all_outputs = []
    #     while self.config['graph_learn'] and (iter_ == 0 or torch.sum(batch_stop_indicators).item() > 0) and iter_ < max_iter_:
    #         iter_ += 1
    #         batch_last_iters += batch_stop_indicators
    #         pre_node_anchor_adj = cur_node_anchor_adj

    #         # Compute n x s node-anchor relationship matrix
    #         cur_node_anchor_adj = network.learn_graph(
    #             network.graph_learner2, node_vec, anchor_features=anchor_vec, node_mask=node_mask, anchor_mask=anchor_mask)

    #         # Compute s x s anchor graph
    #         cur_anchor_adj = compute_anchor_adj(
    #             cur_node_anchor_adj, anchor_mask=anchor_mask)

    #         cur_agg_vec = network.encoder.graph_encoders[0](
    #             init_node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)

    #         update_adj_ratio = self.config.get('update_adj_ratio', None)
    #         if update_adj_ratio is not None:
    #             cur_agg_vec = update_adj_ratio * cur_agg_vec + \
    #                 (1 - update_adj_ratio) * first_init_agg_vec

    #         node_vec = (1 - network.graph_skip_conn) * cur_agg_vec + \
    #             network.graph_skip_conn * init_agg_vec

    #         if network.encoder.graph_encoders[0].bn is not None:
    #             node_vec = network.encoder.graph_encoders[0].compute_bn(
    #                 node_vec)

    #         node_vec = torch.relu(node_vec)
    #         node_vec = F.dropout(node_vec, self.config.get(
    #             'gl_dropout', 0), training=network.training)
    #         anchor_vec = batch_select_from_tensor(
    #             node_vec, sampled_node_idx, max_num_anchors, self.device)

    #         # Add mid GNN layers
    #         for encoder in network.encoder.graph_encoders[1:-1]:
    #             mid_cur_agg_vec = encoder(
    #                 node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
    #             if update_adj_ratio is not None:
    #                 mid_first_agg_vecc = encoder(
    #                     node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)
    #                 mid_cur_agg_vec = update_adj_ratio * mid_cur_agg_vec + \
    #                     (1 - update_adj_ratio) * mid_first_agg_vecc

    #             node_vec = (1 - network.graph_skip_conn) * mid_cur_agg_vec + \
    #                 network.graph_skip_conn * \
    #                 encoder(node_vec, init_adj,
    #                         anchor_mp=False, batch_norm=False)

    #             if encoder.bn is not None:
    #                 node_vec = encoder.compute_bn(node_vec)

    #             node_vec = torch.relu(node_vec)
    #             node_vec = F.dropout(node_vec, self.config.get(
    #                 'gl_dropout', 0), training=network.training)
    #             anchor_vec = batch_select_from_tensor(
    #                 node_vec, sampled_node_idx, max_num_anchors, self.device)

    #         cur_agg_vec = network.encoder.graph_encoders[-1](
    #             node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
    #         if update_adj_ratio is not None:
    #             first_agg_vec = network.encoder.graph_encoders[-1](
    #                 node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)
    #             cur_agg_vec = update_adj_ratio * cur_agg_vec + \
    #                 (1 - update_adj_ratio) * first_agg_vec

    #         tmp_output = (1 - network.graph_skip_conn) * cur_agg_vec + \
    #             network.graph_skip_conn * network.encoder.graph_encoders[-1](
    #                 node_vec, init_adj, anchor_mp=False, batch_norm=False)

    #         tmp_output = network.compute_output(
    #             tmp_output, node_mask=node_mask)
    #         batch_all_outputs.append(tmp_output.unsqueeze(1))

    #         tmp_loss = self.model.criterion(
    #             tmp_output, targets, reduction='none')
    #         if len(tmp_loss.shape) == 2:
    #             tmp_loss = torch.mean(tmp_loss, 1)

    #         loss += batch_stop_indicators.float() * tmp_loss

    #         if self.config['graph_learn'] and self.config['graph_learn_regularization']:
    #             loss += batch_stop_indicators.float() * self.add_batch_graph_loss(cur_anchor_adj,
    #                                                                               raw_anchor_vec, keep_batch_dim=True)

    #         if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
    #             loss += batch_stop_indicators.float() * batch_SquaredFrobeniusNorm(cur_node_anchor_adj -
    #                                                                                pre_node_anchor_adj) * self.config.get('graph_learn_ratio')

    #         tmp_stop_criteria = batch_diff(
    #             cur_node_anchor_adj, pre_node_anchor_adj, cur_node_anchor_adj) > eps_adj
    #         batch_stop_indicators = batch_stop_indicators * tmp_stop_criteria

    #     if iter_ > 0:
    #         loss = torch.mean(loss / batch_last_iters.float()) + loss1

    #         batch_all_outputs = torch.cat(batch_all_outputs, 1)
    #         selected_iter_index = batch_last_iters.long().unsqueeze(-1) - 1

    #         if len(batch_all_outputs.shape) == 3:
    #             selected_iter_index = selected_iter_index.unsqueeze(
    #                 -1).expand(-1, -1, batch_all_outputs.size(-1))
    #             output = batch_all_outputs.gather(
    #                 1, selected_iter_index).squeeze(1)
    #         else:
    #             output = batch_all_outputs.gather(1, selected_iter_index)

    #         score = self.model.score_func(targets.cpu(), output.detach().cpu())

    #     else:
    #         loss = loss1

    #     res = {'loss': loss.item(),
    #            'metrics': {'nloss': -loss.item(), self.model.metric_name: score},
    #            }
    #     if out_predictions:
    #         res['predictions'] = output.detach().cpu()

    #     if training:
    #         # Normalize our loss (if averaged)
    #         loss = loss / self.config['grad_accumulated_steps']
    #         loss.backward()

    #         # Wait for several backward steps
    #         if (step + 1) % self.config['grad_accumulated_steps'] == 0:
    #             self.model.clip_grad()
    #             self.model.optimizer.step()
    #             self.model.optimizer.zero_grad()
    #     return res

    def _run_whole_epoch(self, data_loader, training=True, verbose=None, out_predictions=False):
        '''BP after all iterations'''
        mode = "train" if training else ("test" if self.is_test else "dev")
        self.model.network.train(training)

        init_adj, features, labels = data_loader['adj'], data_loader['features'], data_loader['labels']

        if mode == 'train':
            idx = data_loader['idx_train']
        elif mode == 'dev':
            idx = data_loader['idx_val']
        else:
            idx = data_loader['idx_test']

        network = self.model.network

        # Init
        features = F.dropout(features, network.config.get(
            'feat_adj_dropout', 0), training=network.training)
        init_node_vec = features

        cur_raw_adj, cur_adj = network.learn_graph(
            network.graph_learner, init_node_vec, network.graph_skip_conn, graph_include_self=network.graph_include_self, init_adj=init_adj)
        if self.config['graph_learn'] and self.config.get('max_iter', 10) > 0:
            cur_raw_adj = F.dropout(cur_raw_adj, network.config.get(
                'feat_adj_dropout', 0), training=network.training)
        cur_adj = F.dropout(cur_adj, network.config.get(
            'feat_adj_dropout', 0), training=network.training)

        if network.graph_module == 'gat':
            assert self.config['graph_learn'] is False and self.config.get(
                'max_iter', 10) == 0
            node_vec = network.encoder(init_node_vec, init_adj)
            output = F.log_softmax(node_vec, dim=-1)

        elif network.graph_module == 'graphsage':
            assert self.config['graph_learn'] is False and self.config.get(
                'max_iter', 10) == 0
            # Convert adj to DGLGraph
            import dgl
            from scipy import sparse
            binarized_adj = sparse.coo_matrix(
                init_adj.detach().cpu().numpy() != 0)
            dgl_graph = dgl.DGLGraph(binarized_adj)

            node_vec = network.encoder(dgl_graph, init_node_vec)
            output = F.log_softmax(node_vec, dim=-1)

        else:
            node_vec = torch.relu(
                network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, network.dropout,
                                 training=network.training)

            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(
                    node_vec, network.dropout, training=network.training)

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
            output = F.log_softmax(output, dim=-1)

        score = self.model.score_func(labels[idx], output[idx])
        loss1 = self.model.criterion(output[idx], labels[idx])

        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            loss1 += self.add_graph_loss(cur_raw_adj, init_node_vec)

        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        if not mode == 'test':
            if self._epoch > self.config.get('pretrain_epoch', 0):
                max_iter_ = self.config.get('max_iter', 10)  # Fine-tuning
                if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')

            else:
                max_iter_ = 0  # Pretraining
        else:
            max_iter_ = self.config.get('max_iter', 10)

        if training:
            # cora: 5.5e-8, cora w/o input graph: 1e-8, citeseer w/o input graph: 1e-8, wine: 2e-5, cancer: 2e-5, digtis: 2e-5
            eps_adj = float(self.config.get('eps_adj', 0))
        else:
            eps_adj = float(self.config.get(
                'test_eps_adj', self.config.get('eps_adj', 0)))

        pre_raw_adj = cur_raw_adj
        pre_adj = cur_adj

        loss = 0
        iter_ = 0
        while self.config['graph_learn'] and (iter_ == 0 or diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj) and iter_ < max_iter_:
            iter_ += 1
            pre_adj = cur_adj
            pre_raw_adj = cur_raw_adj
            cur_raw_adj, cur_adj = network.learn_graph(
                network.graph_learner2, node_vec, network.graph_skip_conn, graph_include_self=network.graph_include_self, init_adj=init_adj)

            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_adj = update_adj_ratio * cur_adj + \
                    (1 - update_adj_ratio) * first_adj

            node_vec = torch.relu(
                network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, self.config.get(
                'gl_dropout', 0), training=network.training)

            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, self.config.get(
                    'gl_dropout', 0), training=network.training)

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
            output = F.log_softmax(output, dim=-1)
            score = self.model.score_func(labels[idx], output[idx])
            loss += self.model.criterion(output[idx], labels[idx])

            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                loss += self.add_graph_loss(cur_raw_adj, init_node_vec)

            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += SquaredFrobeniusNorm(cur_raw_adj - pre_raw_adj) * \
                    self.config.get('graph_learn_ratio')

        if mode == 'test' and self.config.get('out_raw_learned_adj_path', None):
            out_raw_learned_adj_path = os.path.join(
                self.dirname, self.config['out_raw_learned_adj_path'])
            np.save(out_raw_learned_adj_path, cur_raw_adj.cpu())
            print('Saved raw_learned_adj to {}'.format(out_raw_learned_adj_path))

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1

        if training:
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.clip_grad()
            self.model.optimizer.step()

        self._update_metrics(loss.item(), {
                             'nloss': -loss.item(), self.model.metric_name: score}, 1, training=training)
        return output[idx], labels[idx]

    # def _scalable_run_whole_epoch(self, data_loader, training=True, verbose=None, out_predictions=False):
    #     '''Scalable run: BP after all iterations'''
    #     mode = "train" if training else ("test" if self.is_test else "dev")
    #     self.model.network.train(training)

    #     init_adj, features, labels = data_loader['adj'], data_loader['features'], data_loader['labels']

    #     if mode == 'train':
    #         idx = data_loader['idx_train']
    #     elif mode == 'dev':
    #         idx = data_loader['idx_val']
    #     else:
    #         idx = data_loader['idx_test']

    #     network = self.model.network

    #     # Init
    #     features = F.dropout(features, network.config.get(
    #         'feat_adj_dropout', 0), training=network.training)
    #     init_node_vec = features

    #     # Randomly sample s anchor nodes
    #     init_anchor_vec, sampled_node_idx = sample_anchors(
    #         init_node_vec, network.config.get('num_anchors', int(0.2 * init_node_vec.size(0))))

    #     # Compute n x s node-anchor relationship matrix
    #     cur_node_anchor_adj = network.learn_graph(
    #         network.graph_learner, init_node_vec, anchor_features=init_anchor_vec)

    #     # Compute s x s anchor graph
    #     cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)

    #     if self.config['graph_learn'] and self.config.get('max_iter', 10) > 0:
    #         cur_node_anchor_adj = F.dropout(cur_node_anchor_adj, network.config.get(
    #             'feat_adj_dropout', 0), training=network.training)

    #     cur_anchor_adj = F.dropout(cur_anchor_adj, network.config.get(
    #         'feat_adj_dropout', 0), training=network.training)

    #     # Update node embeddings via node-anchor-node message passing
    #     init_agg_vec = network.encoder.graph_encoders[0](
    #         init_node_vec, init_adj, anchor_mp=False, batch_norm=False)
    #     node_vec = (1 - network.graph_skip_conn) * network.encoder.graph_encoders[0](init_node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
    #         network.graph_skip_conn * init_agg_vec

    #     if network.encoder.graph_encoders[0].bn is not None:
    #         node_vec = network.encoder.graph_encoders[0].compute_bn(node_vec)

    #     node_vec = torch.relu(node_vec)
    #     node_vec = F.dropout(node_vec, network.dropout,
    #                          training=network.training)
    #     anchor_vec = node_vec[sampled_node_idx]

    #     first_node_anchor_adj, first_anchor_adj = cur_node_anchor_adj, cur_anchor_adj
    #     first_init_agg_vec = network.encoder.graph_encoders[0](
    #         init_node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)

    #     # Add mid GNN layers
    #     for encoder in network.encoder.graph_encoders[1:-1]:
    #         node_vec = (1 - network.graph_skip_conn) * encoder(node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
    #             network.graph_skip_conn * \
    #             encoder(node_vec, init_adj, anchor_mp=False, batch_norm=False)

    #         if encoder.bn is not None:
    #             node_vec = encoder.compute_bn(node_vec)

    #         node_vec = torch.relu(node_vec)
    #         node_vec = F.dropout(node_vec, network.dropout,
    #                              training=network.training)
    #         anchor_vec = node_vec[sampled_node_idx]

    #     # Compute output via node-anchor-node message passing
    #     output = (1 - network.graph_skip_conn) * network.encoder.graph_encoders[-1](node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
    #         network.graph_skip_conn * network.encoder.graph_encoders[-1](
    #             node_vec, init_adj, anchor_mp=False, batch_norm=False)
    #     output = F.log_softmax(output, dim=-1)
    #     score = self.model.score_func(labels[idx], output[idx])
    #     loss1 = self.model.criterion(output[idx], labels[idx])

    #     if self.config['graph_learn'] and self.config['graph_learn_regularization']:
    #         loss1 += self.add_graph_loss(cur_anchor_adj, init_anchor_vec)

    #     if not mode == 'test':
    #         if self._epoch > self.config.get('pretrain_epoch', 0):
    #             max_iter_ = self.config.get('max_iter', 10)  # Fine-tuning
    #             if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
    #                 for k in self._dev_metrics:
    #                     self._best_metrics[k] = -float('inf')

    #         else:
    #             max_iter_ = 0  # Pretraining
    #     else:
    #         max_iter_ = self.config.get('max_iter', 10)

    #     if training:
    #         # cora: 5.5e-8, cora w/o input graph: 1e-8, citeseer w/o input graph: 1e-8, wine: 2e-5, cancer: 2e-5, digtis: 2e-5
    #         eps_adj = float(self.config.get('eps_adj', 0))
    #     else:
    #         eps_adj = float(self.config.get(
    #             'test_eps_adj', self.config.get('eps_adj', 0)))

    #     pre_node_anchor_adj = cur_node_anchor_adj

    #     loss = 0
    #     iter_ = 0
    #     while self.config['graph_learn'] and (iter_ == 0 or diff(cur_node_anchor_adj, pre_node_anchor_adj, cur_node_anchor_adj).item() > eps_adj) and iter_ < max_iter_:
    #         iter_ += 1
    #         pre_node_anchor_adj = cur_node_anchor_adj

    #         # Compute n x s node-anchor relationship matrix
    #         cur_node_anchor_adj = network.learn_graph(
    #             network.graph_learner2, node_vec, anchor_features=anchor_vec)

    #         # Compute s x s anchor graph
    #         cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)

    #         cur_agg_vec = network.encoder.graph_encoders[0](
    #             init_node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)

    #         update_adj_ratio = self.config.get('update_adj_ratio', None)
    #         if update_adj_ratio is not None:
    #             cur_agg_vec = update_adj_ratio * cur_agg_vec + \
    #                 (1 - update_adj_ratio) * first_init_agg_vec

    #         node_vec = (1 - network.graph_skip_conn) * cur_agg_vec + \
    #             network.graph_skip_conn * init_agg_vec

    #         if network.encoder.graph_encoders[0].bn is not None:
    #             node_vec = network.encoder.graph_encoders[0].compute_bn(
    #                 node_vec)

    #         node_vec = torch.relu(node_vec)
    #         node_vec = F.dropout(node_vec, self.config.get(
    #             'gl_dropout', 0), training=network.training)
    #         anchor_vec = node_vec[sampled_node_idx]

    #         # Add mid GNN layers
    #         for encoder in network.encoder.graph_encoders[1:-1]:
    #             mid_cur_agg_vec = encoder(
    #                 node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
    #             if update_adj_ratio is not None:
    #                 mid_first_agg_vecc = encoder(
    #                     node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)
    #                 mid_cur_agg_vec = update_adj_ratio * mid_cur_agg_vec + \
    #                     (1 - update_adj_ratio) * mid_first_agg_vecc

    #             node_vec = (1 - network.graph_skip_conn) * mid_cur_agg_vec + \
    #                 network.graph_skip_conn * \
    #                 encoder(node_vec, init_adj,
    #                         anchor_mp=False, batch_norm=False)

    #             if encoder.bn is not None:
    #                 node_vec = encoder.compute_bn(node_vec)

    #             node_vec = torch.relu(node_vec)
    #             node_vec = F.dropout(node_vec, self.config.get(
    #                 'gl_dropout', 0), training=network.training)
    #             anchor_vec = node_vec[sampled_node_idx]

    #         cur_agg_vec = network.encoder.graph_encoders[-1](
    #             node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
    #         if update_adj_ratio is not None:
    #             first_agg_vec = network.encoder.graph_encoders[-1](
    #                 node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)
    #             cur_agg_vec = update_adj_ratio * cur_agg_vec + \
    #                 (1 - update_adj_ratio) * first_agg_vec

    #         output = (1 - network.graph_skip_conn) * cur_agg_vec + \
    #             network.graph_skip_conn * network.encoder.graph_encoders[-1](
    #                 node_vec, init_adj, anchor_mp=False, batch_norm=False)

    #         output = F.log_softmax(output, dim=-1)
    #         score = self.model.score_func(labels[idx], output[idx])
    #         loss += self.model.criterion(output[idx], labels[idx])

    #         if self.config['graph_learn'] and self.config['graph_learn_regularization']:
    #             loss += self.add_graph_loss(cur_anchor_adj, init_anchor_vec)

    #         if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
    #             loss += SquaredFrobeniusNorm(cur_node_anchor_adj -
    #                                          pre_node_anchor_adj) * self.config.get('graph_learn_ratio')

    #     if mode == 'test' and self.config.get('out_raw_learned_adj_path', None):
    #         out_raw_learned_adj_path = os.path.join(
    #             self.dirname, self.config['out_raw_learned_adj_path'])
    #         np.save(out_raw_learned_adj_path, cur_node_anchor_adj.cpu())
    #         print('Saved raw_learned_adj to {}'.format(out_raw_learned_adj_path))

    #     if iter_ > 0:
    #         loss = loss / iter_ + loss1
    #     else:
    #         loss = loss1
    #     if training:
    #         self.model.optimizer.zero_grad()
    #         loss.backward()
    #         self.model.clip_grad()
    #         self.model.optimizer.step()

    #     self._update_metrics(loss.item(), {
    #                          'nloss': -loss.item(), self.model.metric_name: score}, 1, training=training)
    #     return output[idx], labels[idx]

    def _run_batch_epoch(self, data_loader, training=True, rl_ratio=0, verbose=10, out_predictions=False):
        start_time = time.time()
        mode = "train" if training else ("test" if self.is_test else "dev")

        if training:
            self.model.optimizer.zero_grad()
        output = []
        gold = []
        for step in range(data_loader.get_num_batch()):
            input_batch = data_loader.nextBatch()
            x_batch = vectorize_input(
                input_batch, self.config, training=training, device=self.device)
            if not x_batch:
                continue  # When there are no examples in the batch

            if self.config.get('no_gnn', False):
                res = self.batch_no_gnn(
                    x_batch, step, training=training, out_predictions=out_predictions)
            else:
                if self.config.get('scalable_run', False):
                    res = self.scalable_batch_IGL_stop(
                        x_batch, step, training=training, out_predictions=out_predictions)
                else:
                    res = self.batch_IGL_stop(
                        x_batch, step, training=training, out_predictions=out_predictions)

            loss = res['loss']
            metrics = res['metrics']
            self._update_metrics(
                loss, metrics, x_batch['batch_size'], training=training)

            if training:
                self._n_train_examples += x_batch['batch_size']

            if (verbose > 0) and (step > 0) and (step % verbose == 0):
                summary_str = self.self_report(step, mode)
                self.logger.write_to_file(summary_str)
                print(summary_str)
                print('used_time: {:0.2f}s'.format(time.time() - start_time))

            if not training and out_predictions:
                output.extend(res['predictions'])
                gold.extend(x_batch['targets'])
        return output, gold

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def self_report(self, step, mode='train'):
        if mode == "train":
            format_str = "[train-{}] step: [{} / {}] | loss = {:0.5f}".format(
                self._epoch, step, self._n_train_batches, self._train_loss.mean())
            format_str += self.metric_to_str(self._train_metrics)
        elif mode == "dev":
            format_str = "[predict-{}] step: [{} / {}] | loss = {:0.5f}".format(
                self._epoch, step, self._n_dev_batches, self._dev_loss.mean())
            format_str += self.metric_to_str(self._dev_metrics)
        elif mode == "test":
            format_str = "[test] | test_exs = {} | step: [{} / {}]".format(
                self._n_test_examples, step, self._n_test_batches)
            format_str += self.metric_to_str(self._dev_metrics)
        else:
            raise ValueError('mode = {} not supported.' % mode)
        return format_str

    def plain_metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k])
        return format_str

    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(
                k.upper(), metrics[k].mean())
        return format_str

    def best_metric_to_str(self, metrics):
        format_str = '\n'
        for k in metrics:
            format_str += '{} = {:0.5f}\n'.format(k.upper(), metrics[k])
        return format_str

    def summary(self):
        start = "\n<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        info = "Best epoch = {}; ".format(
            self._best_epoch) + self.best_metric_to_str(self._best_metrics)
        end = " <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        return "\n".join([start, info, end])

    def _update_metrics(self, loss, metrics, batch_size, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
            for k in self._train_metrics:
                if not k in metrics:
                    continue
                self._train_metrics[k].update(metrics[k], batch_size)
        else:
            if loss:
                self._dev_loss.update(loss)
            for k in self._dev_metrics:
                if not k in metrics:
                    continue
                self._dev_metrics[k].update(metrics[k], batch_size)

    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

        for k in self._train_metrics:
            self._train_metrics[k].reset()
        for k in self._dev_metrics:
            self._dev_metrics[k].reset()

    def _stop_condition(self, epoch, patience=10):
        """
        Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True

    def add_graph_loss(self, out_adj, features):
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.config['smoothness_ratio'] * torch.trace(torch.mm(
            features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = to_cuda(torch.ones(out_adj.size(-1)), self.device)
        graph_loss += -self.config['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(
            out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze() / out_adj.shape[-1]
        graph_loss += self.config['sparsity_ratio'] * \
            torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss

    def add_batch_graph_loss(self, out_adj, features, keep_batch_dim=False):
        # Graph regularization
        if keep_batch_dim:
            graph_loss = []
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss.append(self.config['smoothness_ratio'] * torch.trace(torch.mm(
                    features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape[1:])))

            graph_loss = to_cuda(torch.Tensor(graph_loss), self.device)

            ones_vec = to_cuda(torch.ones(out_adj.shape[:-1]), self.device)
            graph_loss += -self.config['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(torch.matmul(
                out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze(-1).squeeze(-1) / out_adj.shape[-1]
            graph_loss += self.config['sparsity_ratio'] * torch.sum(
                torch.pow(out_adj, 2), (1, 2)) / int(np.prod(out_adj.shape[1:]))

        else:
            graph_loss = 0
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss += self.config['smoothness_ratio'] * torch.trace(torch.mm(
                    features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape))

            ones_vec = to_cuda(torch.ones(out_adj.shape[:-1]), self.device)
            graph_loss += -self.config['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(torch.matmul(
                out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).sum() / out_adj.shape[0] / out_adj.shape[-1]
            graph_loss += self.config['sparsity_ratio'] * torch.sum(
                torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss


def diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    return diff_


def batch_diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2), (1, 2))  # Shape: [batch_size]
    norm_ = torch.sum(torch.pow(Z, 2), (1, 2))
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    return diff_


def SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))


def batch_SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2), (1, 2)) / int(np.prod(X.shape[1:]))
