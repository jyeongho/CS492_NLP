# coding=utf-8

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
from tqdm import tqdm, trange
from transformers.tokenization_bert import whitespace_tokenize
from transformers import (
    AdamW,
    ElectraTokenizer,
    get_linear_schedule_with_warmup,
)
from model_electra import ElectraForSequenceClassification
from open_squad import squad_convert_examples_to_features

'''
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
'''
# ''
# KorQuAD-Open-Naver-Search 사용할때 전처리 코드.
from open_squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from open_squad import SquadResult, SquadV1Processor, SquadV2Processor

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


def _infer(model, tokenizer, my_args, root_path):
    my_args.data_dir = root_path
    _, predictions = predict(my_args, model, tokenizer, val_or_test="test")
    qid_and_answers = [("test-{}".format(qid), answer) for qid, (_, answer) in enumerate(predictions.items())]
    return qid_and_answers


def bind_nsml(model, tokenizer, my_args):

    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(dir_name, 'model.pt'))
        torch.save(tokenizer, os.path.join(dir_name, 'tokenizer'))
        torch.save(my_args, os.path.join(dir_name, "my_args.bin"))

        logger.info("Save model & tokenizer & args at {}".format(dir_name))

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        model.load_state_dict(state)

        temp_my_args = torch.load(os.path.join(dir_name, "my_args.bin"))
        nsml.copy(temp_my_args, my_args)

        temp_tokenizer = torch.load(os.path.join(dir_name, 'tokenizer'))
        nsml.copy(temp_tokenizer, tokenizer)

        logger.info("Load model & tokenizer & args from {}".format(dir_name))

    def infer(root_path):
        """NSML will record f1-score based on predictions from this method."""
        result = _infer(model, tokenizer, my_args, root_path)
        for line in result:
            assert type(tuple(line)) == tuple and len(line) == 2, "Wrong infer result: {}".format(line)
        return result

    nsml.bind(save=save, load=load, infer=infer)

'''
Train Sketchy Reading Module of retrospective paper.
Using ElectraForSequenceClassification model, 
this function does training by classification task.
After training, the output of ElectraForSequenceClassification model will be used to calculating ext score.
Reference : https://github.com/cooelf/AwesomeMRC/blob/master/transformer-mrc/examples/run_cls.py
'''
def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    
    tr_loss, logging_loss = 0.0, 0.0
    tr_acc, logging_acc = 0.0, 0.0
    best_eval_acc = 0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproducibility
    set_seed(args)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True, disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[7],
            }

            outputs = model(**inputs)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            gt = inputs["labels"].detach().cpu().numpy()
            acc = (preds == gt).mean()
            

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            tr_acc += acc

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.evaluate_during_training:
                        logger.info("Validation start for epoch {}".format(epoch))
                        eval_loss, eval_acc = evaluate(args, model, tokenizer, prefix=epoch)
                        is_best = eval_acc > best_eval_acc
                        best_eval_acc = max(eval_acc, best_eval_acc)

                        current_loss = (tr_loss - logging_loss) / args.logging_steps
                        logging_loss = tr_loss
                        current_acc = (tr_acc - logging_acc) / args.logging_steps
                        logging_acc = tr_acc

                        logger.info(
                            "best_eval_acc = {}, eval_acc = {}, eval_loss = {}, acc = {}, loss = {}, global_step = {}, " \
                            .format(best_eval_acc, eval_acc, eval_loss, current_acc, current_loss, global_step))
                        if IS_ON_NSML:
                            nsml.report(summary=True, step=global_step, eval_acc=eval_acc, eval_loss=eval_loss, acc=current_acc, loss=current_loss)
                            if is_best:
                                nsml.save(args.model_type + "_best")

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if IS_ON_NSML:
                        nsml.save(args.model_type + "_gs{}_e{}".format(global_step, epoch))
                    else:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if IS_ON_NSML:
        nsml.save(args.model_type + "_last")

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", val_or_test="val"):
    # Compute the eval_loss and acc.
    eval_loss, acc = predict(args, model, tokenizer, prefix=prefix, val_or_test=val_or_test)
    return eval_loss, acc


def predict(args, model, tokenizer, prefix="", val_or_test="val"):
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
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = timeit.default_timer()

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[6]
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        
        logits = logits.detach().cpu().numpy()
        nb_eval_steps += 1
        if preds is None:
            preds = logits
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits, axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy())
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    acc = (preds == out_label_ids).mean()

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    return eval_loss, acc


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, val_or_test="val"):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache.
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."

    # Init features and dataset
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

    #if args.local_rank in [-1, 0]:
    #    logger.info("Saving features into cached file %s", cached_features_file)
    #    torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

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
        "--model_type",
        default=None,
        type=str,
        required=True,
    )
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
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
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
    model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v3-discriminator")

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model, tokenizer, args)
        if args.pause:
            nsml.paused(scope=locals())
    ################################

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

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
