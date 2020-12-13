# coding=utf-8

"""
Predict predictions using Classification model and QuestionAnswering model.
Use ElectraForSequenceClassification model as sketchy reading module.
Use ElectraForQuestionAnswering model as intensive reading module.
Since this file is only for making prediction, there is no train function.

"""

import argparse
import logging
import os
import random
import timeit
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import ElectraTokenizer
from model_electra import ElectraForSequenceClassification, ElectraForQuestionAnswering
from open_squad import squad_convert_examples_to_features
from scipy.special import softmax

# ''
# KorQuAD-Open-Naver-Search 사용할때 전처리 코드.
from open_squad_metrics import (
    squad_evaluate,
    compute_predictions_logits_with_score
)
from open_squad import SquadResult, SquadV1Processor, SquadV2Processor, SquadResult_with_score

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

def _infer(model_sc, model_qa, tokenizer, my_args, root_path):
    my_args.data_dir = root_path
    _, predictions = predict(my_args, model_sc, model_qa, tokenizer, val_or_test="test")
    qid_and_answers = [("test-{}".format(qid), answer) for qid, (_, answer) in enumerate(predictions.items())]
    return qid_and_answers


def bind_nsml(model_sc, model_qa, tokenizer, my_args):

    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)

        torch.save(model_sc.state_dict(), os.path.join(dir_name, 'model_sc.pt'))
        torch.save(model_qa.state_dict(), os.path.join(dir_name, 'model_qa.pt'))
        logger.info("Save model at {}".format(dir_name))

    def load(dir_name, *args, **kwargs):
        state_sc = torch.load(os.path.join(dir_name, 'model_sc.pt'))
        model_sc.load_state_dict(state_sc)

        state_qa = torch.load(os.path.join(dir_name, 'model_qa.pt'))
        model_qa.load_state_dict(state_qa)

        logger.info("Load model from {}".format(dir_name))

    def infer(root_path):
        """NSML will record f1-score based on predictions from this method."""
        result = _infer(model_sc, model_qa, tokenizer, my_args, root_path)
        for line in result:
            assert type(tuple(line)) == tuple and len(line) == 2, "Wrong infer result: {}".format(line)
        return result

    nsml.bind(save=save, load=load, infer=infer)


def evaluate(args, model_sc, model_qa, tokenizer, prefix="", val_or_test="val"):
    examples, predictions = predict(args, model_sc, model_qa, tokenizer, prefix=prefix, val_or_test=val_or_test)
    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def predict(args, model_sc, model_qa, tokenizer, prefix="", val_or_test="val"):
    dataset, examples, features = load_and_cache_examples(
        args, tokenizer, evaluate=True, output_examples=True,
        val_or_test=val_or_test,
    )

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model_sc, torch.nn.DataParallel):
        model_sc = torch.nn.DataParallel(model_sc)
        model_qa = torch.nn.DataParallel(model_qa)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True):
        model_sc.eval()
        model_qa.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs_sc = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[6],
            }

            inputs_qa = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "pq_end_pos": batch[7]
            }

            example_indices = batch[3]

            outputs_sc = model_sc(**inputs_sc)
            outputs_qa = model_qa(**inputs_qa)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            ext_logits = outputs_sc[1][i].detach().cpu().numpy().reshape(1, 2)
            output_qa = [to_list(output[i]) for output in outputs_qa]

            ext_logits = softmax(ext_logits).squeeze()
            score_ext = ext_logits[1] - ext_logits[0]

            start_logits, end_logits = output_qa
            start_logits = softmax(start_logits).squeeze().tolist()
            end_logits = softmax(end_logits).squeeze().tolist()
            result = SquadResult_with_score(unique_id, start_logits, end_logits, score_ext, label=inputs_sc["labels"][i].detach().cpu().item())

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    is_test = (val_or_test == "test")
    
    if is_test:
        predictions = compute_predictions_logits_with_score(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
            is_test=is_test,
        )
    else:
        predictions, label_v = compute_predictions_logits_with_score(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
            is_test=is_test,
        )
        print('len of label 0: {}'.format(len(label_v[0])))
        print('len of label 1: {}'.format(len(label_v[1])))
        bins = np.arange(-3, 3.1, 0.1)
        label_0_hist, bins = np.histogram(label_v[0], bins)
        label_1_hist, bins = np.histogram(label_v[1], bins)

        for i in range(len(bins) - 1):
            print('bins: {:.2f}~{:.2f}, label_0_hist: {}, label_1_hist: {}'.format(bins[i], bins[i+1], label_0_hist[i], label_1_hist[i]))
    

    return examples, predictions


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, val_or_test="val"):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache.
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."

    # Init features and dataset from cache if it exists
    
    logger.info("Creating features from dataset file at %s", input_dir)

    if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

        if args.version_2_with_negative:
            logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

        tfds_examples = tfds.load("squad")
        examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
    else:
        processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
        if evaluate:
            filename = args.predict_file if val_or_test == "val" else "test_data/korquad_open_test.json"
            examples = processor.get_eval_examples(args.data_dir, filename=filename)
        else:
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

    print("Starting squad_convert_examples_to_features")
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=args.threads,
    )
    print("Complete squad_convert_examples_to_features")


    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache.
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", default=True,
        action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=24, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
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

    # for NSML
    args.data_dir = os.path.join(DATASET_PATH, args.data_dir)

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

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

    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model_SC, model_QA, tokenizer, args)
        if args.pause:
            nsml.paused(scope=locals())
    ################################
    
    #Before loading, save models using 'run_save_model.py' to gather models in separate sessions.
    nsml.load(checkpoint='saved', session="kaist006/korquad-open-ldbd3/160")
    nsml.save('best_model')

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is
    # set. Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running
    # `--fp16_opt_level="O2"` will remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    if args.do_eval:
        result = evaluate(args, model_SC, model_QA, tokenizer)
        _f1, _exact = result["f1"], result["exact"]
        print('f1: {}, exact: {}'.format(_f1, _exact))


if __name__ == "__main__":
    main()
