###########################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###########################################################################
import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
import ast

from flowtron import FlowtronLoss
from flowtron import Flowtron
from data import Data, DataCollate
from flowtron_logger import FlowtronLogger


#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

def update_params(config, params):
    for param in params:
        print(param)
        k, v = param.split("=")
        try:
            v = ast.literal_eval(v)
        except:
            pass

        k_split = k.split('.')
        if len(k_split) > 1:
            parent_k = k_split[0]
            cur_param = ['.'.join(k_split[1:])+"="+str(v)]
            update_params(config[parent_k], cur_param)
        elif k in config and len(k_split) == 1:
            config[k] = v
        else:
            print("{}, {} params not updated".format(k, v))


def prepare_dataloaders(data_config, n_gpus, batch_size):
    # Get data, data loaders and 1ollate function ready
    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(data_config['training_files'],
                    **dict((k, v) for k, v in data_config.items()
                    if k not in ignore_keys))

    valset = Data(data_config['validation_files'],
                  **dict((k, v) for k, v in data_config.items()
                  if k not in ignore_keys), speaker_ids=trainset.speaker_ids)

    collate_fn = DataCollate()

    train_sampler, shuffle = None, True
    if n_gpus > 1:
        train_sampler, shuffle = DistributedSampler(trainset), False

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler, batch_size=batch_size,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    return train_loader, valset, collate_fn


def warmstart(checkpoint_path, model, include_layers=None):
    print("Warm starting model", checkpoint_path)
    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in pretrained_dict:
        pretrained_dict = pretrained_dict['model'].state_dict()
    else:
        pretrained_dict = pretrained_dict['state_dict']

    if include_layers is not None:
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if any(l in k for l in include_layers)}

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict}

    if pretrained_dict['speaker_embedding.weight'].shape != model_dict['speaker_embedding.weight'].shape:
        del pretrained_dict['speaker_embedding.weight']

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer, ignore_layers=[]):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    model_dict = checkpoint_dict['model'].state_dict()

    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    else:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    model.load_state_dict(model_dict)
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = Flowtron(**model_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def compute_validation_loss(model, criterion, valset, collate_fn, batch_size,
                            n_gpus):
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if n_gpus > 1 else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            mel, speaker_vecs, text, in_lens, out_lens, gate_target = batch
            mel, speaker_vecs, text = mel.cuda(), speaker_vecs.cuda(), text.cuda()
            in_lens, out_lens, gate_target = in_lens.cuda(), out_lens.cuda(), gate_target.cuda()
            z, log_s_list, gate_pred, attn, mean, log_var, prob = model(
                mel, speaker_vecs, text, in_lens, out_lens)

            loss = criterion((z, log_s_list, gate_pred, mean, log_var, prob),
                             gate_target, out_lens)

            if n_gpus > 1:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    print("Mean {}\nLogVar {}\nProb {}".format(mean, log_var, prob))
    model.train()
    return val_loss, attn, gate_pred, gate_target


def train(n_gpus, rank, output_directory, epochs, learning_rate, weight_decay,
          sigma, iters_per_checkpoint, batch_size, seed, checkpoint_path,
          ignore_layers, include_layers, warmstart_checkpoint_path,
          with_tensorboard, fp16_run):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if n_gpus > 1:
        init_distributed(rank, n_gpus, **dist_config)

    criterion = FlowtronLoss(sigma, model_config['n_components'] > 1,
                             model_config['use_gate_layer'])
    model = Flowtron(**model_config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    # Load checkpoint if one exists
    iteration = 0
    if warmstart_checkpoint_path != "":
        model = warmstart(warmstart_checkpoint_path, model, include_layers)

    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                                      optimizer, ignore_layers)
        iteration += 1  # next iteration is iteration + 1

    if n_gpus > 1:
        model = apply_gradient_allreduce(model)
    print(model)
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    train_loader, valset, collate_fn = prepare_dataloaders(
        data_config, n_gpus, batch_size)

    # Get shared output_directory ready
    if rank == 0 and not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory)

    if with_tensorboard and rank == 0:
        logger = FlowtronLogger(os.path.join(output_directory, 'logs'))

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for batch in train_loader:
            model.zero_grad()

            mel, speaker_vecs, text, in_lens, out_lens, gate_target = batch
            mel, speaker_vecs, text = mel.cuda(), speaker_vecs.cuda(), text.cuda()
            in_lens, out_lens, gate_target = in_lens.cuda(), out_lens.cuda(), gate_target.cuda()

            z, log_s_list, gate_pred, attn, mean, log_var, prob = model(
                mel, speaker_vecs, text, in_lens, out_lens)
            loss = criterion((z, log_s_list, gate_pred, mean, log_var, prob),
                             gate_target, out_lens)

            if n_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            if rank == 0:
                print("{}:\t{:.9f}".format(iteration, reduced_loss), flush=True)

            if with_tensorboard and rank == 0:
                logger.add_scalar('training_loss', reduced_loss, iteration)
                logger.add_scalar('learning_rate', learning_rate, iteration)

            if (iteration % iters_per_checkpoint == 0):
                val_loss, attns, gate_pred, gate_target = compute_validation_loss(
                    model, criterion, valset, collate_fn, batch_size, n_gpus)
                if rank == 0:
                    print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
                    if with_tensorboard:
                        logger.log_validation(
                            val_loss, attns, gate_pred, gate_target, iteration)

                    checkpoint_path = "{}/model_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    args = parser.parse_args()
    args.rank = 0

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)
    print(config)

    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global model_config
    model_config = config["model_config"]

    # Make sure the launcher sets `RANK` and `WORLD_SIZE`.
    rank = int(os.getenv('RANK', '0'))
    n_gpus = int(os.getenv("WORLD_SIZE", '1'))
    print('> got rank {} and world size {} ...'.format(rank, n_gpus))

    if n_gpus == 1 and rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(n_gpus, rank, **train_config)
