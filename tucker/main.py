from load_data import Data
import logging
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import os
import json


model_mapping = {
    "tucker": TuckER,
    "tucker_literal": TuckER_Literal,
    "tucker_kbln": TuckER_KBLN,
    "conve": ConvE,
    "conve_literal": ConvE_Literal,
    "conve_kbln": ConvE_KBLN,
    "distmult": DistMult,
    "distmult_literal": DistMult_Literal,
    "distmult_kbln": DistMult_KBLN,
    "complex": ComplEx,
    "complex_literal": ComplEx_Literal,
    "complex_kbln": ComplEx_KBLN
}


class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 feature_map_dropout=0.2, embedding_shape1=20,
                 hidden_size=10368, use_bias=True,
                 label_smoothing=0.):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2, "feature_map_dropout": feature_map_dropout,
                       "hidden_size": hidden_size, "use_bias": use_bias, "embedding_shape1": embedding_shape1,
                       "dataset": dataset, "ent2idx": self.entity_idxs, "rel2idx": self.relation_idxs,
                       "device": "cuda" if torch.cuda.is_available() else "cpu"}
        # Set up log file
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        # If need checkpoint, save it
        if save_best:
            self.target = f"{args.output}/{dataset}"
            try:
                os.mkdir(self.target)
            except FileExistsError:
                pass
            with open(f"{self.target}/{args.model}_entities.dict", 'w+') as fd:
                json.dump(self.entity_idxs, fd, indent=2)
            with open(f"{self.target}/{args.model}_relations.dict", 'w+') as fd:
                json.dump(self.relation_idxs, fd, indent=2)

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]],
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        logging.info("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        logging.info('Hits @10: {0}'.format(np.mean(hits[9])))
        logging.info('Hits @3: {0}'.format(np.mean(hits[2])))
        logging.info('Hits @1: {0}'.format(np.mean(hits[0])))
        logging.info('Mean rank: {0}'.format(np.mean(ranks)))
        logging.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
        return np.mean(1. / np.array(ranks))

    def train_and_eval(self, model="tucker"):
        logging.info(f"Training the {model} model...")

        train_data_idxs = self.get_data_idxs(d.train_data)
        logging.info("Number of training data points: %d" % len(train_data_idxs))

        model = model_mapping[model](d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
            model.to_cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        logging.info("Starting training...")
        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0])
                r_idx = torch.tensor(data_batch[:, 1])
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            logging.info(f"Step at {it}")
            logging.info(f"Training Time elapsed: {time.time() - start_train}")
            logging.info(f"Training Loss: {np.mean(losses)}")

            model.eval()
            with torch.no_grad():
                if it >= args.warm_up and not it % eval_step:

                    logging.info(f"Validation at step {it}")
                    mrrs.append(self.evaluate(model, d.valid_data))
                    logging.info(f"Test at step {it}")
                    start_test = time.time()
                    self.evaluate(model, d.test_data)
                    logging.info(f"Test Evaluation Time: {time.time() - start_test}")

                    patience = patience - 1 if mrrs[-1] != max(mrrs) else args.patience
                    if save_best and mrrs[-1] == max(mrrs):
                        torch.save(model.state_dict(), f"{self.target}/{args.model}.model")
                    if use_stopper and patience <= 0:
                        logging.info(f"Early stop since no further improvement is made on the validation set")
                        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fb15k237", nargs="?",
                        help="Which dataset to use: fb15k, fb15k237, wn18 or wn18rr.")
    parser.add_argument("--model", type=str, default="tucker", nargs="?",
                        help="Which model to use?")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                        help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                        help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                        help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                        help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                        help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                        help="Dropout after the second hidden layer.")
    parser.add_argument("--feature_map_dropout", type=float, default=0.2, nargs="?",
                        help="Dropout after the feature map (ConvE).")
    parser.add_argument("--hidden_size", type=float, default=9728, nargs="?",
                        help="hidden_size (ConvE).")
    parser.add_argument('--embedding-shape1', type=int, default=20,
                        help='The first dimension of the reshaped 2D embedding. '
                             'The second dimension is infered. Default: 20 (ConvE)')
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                        help="Amount of label smoothing.")
    parser.add_argument('--use_bias', action='store_true',
                        help='Use a bias in the convolutional layer (ConvE). Default: True')
    parser.add_argument("--eval_step", type=int, default=10, nargs="?",
                        help="Evaluation step.")
    parser.add_argument("--input", type=str, default="data", help="input path")
    parser.add_argument("--output", type=str, default="out", help="output path")
    parser.add_argument("--use_stopper", action='store_true', help='Use an early stopper')
    parser.add_argument("--save_best", action='store_true', help='Save best model')
    parser.add_argument("--warm_up", type=int, default=30, nargs="?", help="Grace Period before evaluation")
    parser.add_argument("--patience", type=int, default=5, nargs="?", help="Early Stopper")

    args = parser.parse_args()
    # training parameters
    dataset = args.dataset
    model = args.model
    eval_step = args.eval_step
    # stopper parameters
    use_stopper = args.use_stopper
    save_best = args.save_best
    patience = args.patience
    mrrs = []

    log_file = f"{args.output}/{dataset}_{model}.log"
    data_dir = f"{args.input}/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir, reverse=True)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr,
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1,
                            hidden_dropout2=args.hidden_dropout2, feature_map_dropout=args.feature_map_dropout,
                            embedding_shape1=args.embedding_shape1,
                            hidden_size=args.hidden_size, use_bias=args.use_bias, label_smoothing=args.label_smoothing)
    experiment.train_and_eval(model=model)
