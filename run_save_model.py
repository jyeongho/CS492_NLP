# coding=utf-8
'''
Thie file is for saving models in separate sessions into one session.
Because Sketchy Reading Module and Intensive Reading Module are saved in different session, 
we need to save these model in one session for convenience.
'''
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
from transformers import ElectraTokenizer
from model_electra import ElectraForSequenceClassification, ElectraForQuestionAnswering

import nsml
from nsml import DATASET_PATH, IS_ON_NSML
if not IS_ON_NSML:
    DATASET_PATH = "../korquad-open-ldbd3/"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


# NSML functions
def bind_nsml(model_sc, model_qa, model_type, tokenizer, my_args):

    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)

        torch.save(model_sc.state_dict(), os.path.join(dir_name, 'model_sc.pt'))
        torch.save(model_qa.state_dict(), os.path.join(dir_name, 'model_qa.pt'))
        logger.info("Save model & tokenizer & args at {}".format(dir_name))


    def load_sc(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        model_sc.load_state_dict(state)
        logger.info('SC model loaded.')

    def load_qa(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        model_qa.load_state_dict(state)
        logger.info('qa model loaded.')
    
    if model_type == 'sc':
        nsml.bind(save=save, load=load_sc)
    else:
        nsml.bind(save=save, load=load_qa)


def main():
    parser = argparse.ArgumentParser()

    # Other parameters
    
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
   
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    ### DO NOT MODIFY THIS BLOCK ###
    # arguments for nsml
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    ################################

    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        filename='log.log'
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
    model_SC = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v3-discriminator")
    model_QA = ElectraForQuestionAnswering.from_pretrained("monologg/koelectra-small-v3-discriminator")

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model_SC.to(args.device)
    model_QA.to(args.device)

    # First, Load two models one-by-one.
    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model_SC, model_QA, 'sc', tokenizer, args)
        if args.pause:
            nsml.paused(scope=locals())
    ################################
    nsml.load(checkpoint='electra_best', session="kaist006/korquad-open-ldbd3/121")

    if IS_ON_NSML:
        bind_nsml(model_SC, model_QA, 'qa', tokenizer, args)
        if args.pause:
            nsml.paused(scope=locals())
    ################################
    nsml.load(checkpoint='bert_best', session="kaist006/korquad-open-ldbd3/120")
    
    # Next, save models.
    nsml.save('saved')
    logger.info("Training/evaluation parameters %s", args)


if __name__ == "__main__":
    main()
