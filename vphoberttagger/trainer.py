import transformers

from vphoberttagger.constant import LOGGER, MODEL_MAPPING, LABEL_MAPPING
from vphoberttagger.helper import set_ramdom_seed, plot_confusion_matrix, get_total_model_parameters
from vphoberttagger.arguments import get_train_argument, get_test_argument
from vphoberttagger.dataset import build_dataset
from vphoberttagger.conlleval import evaluate

from tqdm import tqdm
from pathlib import Path
from typing import Union
from prettytable import PrettyTable
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
from torch.utils.data import RandomSampler, DataLoader
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, RobertaForSequenceClassification

import os
import sys
import torch
import time
import datetime
import itertools


def save_model(args, saved_file, model):
    saved_data = {
        'model': model.state_dict(),
        'classes': args.label2id,
        'args': args
    }
    torch.save(saved_data, saved_file)


def validate(model, task, iterator, cur_epoch: int, output_dir: Union[str, os.PathLike] = './', is_test=False):
    start_time = time.time()
    model.eval()
    eval_loss = 0.0
    eval_golds, eval_preds = [], []
    # Run one step on sub-dataset
    with torch.no_grad():
        tqdm_desc = f'[EVAL- Epoch {cur_epoch}]'
        eval_bar = tqdm(enumerate(iterator), total=len(iterator), desc=tqdm_desc)
        for idx, batch in eval_bar:
            outputs = model(**batch)
            eval_loss += outputs.loss.detach().item()
            active_accuracy = batch['label_masks'].view(-1) != 0
            labels = torch.masked_select(batch['labels'].view(-1), active_accuracy)
            eval_golds.extend(labels.detach().cpu().tolist())
            if isinstance(outputs.tags[-1], list):
                eval_preds.extend(list(itertools.chain(*outputs.tags)))
            else:
                eval_preds.extend(outputs.tags)
    epoch_loss = eval_loss / len(iterator)
    if is_test:
        evaluate([LABEL_MAPPING[task]["id2label"][tag] for tag in eval_golds],
                 [LABEL_MAPPING[task]["id2label"][tag] for tag in eval_preds])
        reports: dict = classification_report(eval_golds, eval_preds,
                                              output_dict=False,
                                              zero_division=0,
                                              digits=4,
                                              target_names=LABEL_MAPPING[task]["label2id"])
        LOGGER.info(reports)
        label_index_to_print = list(range(len(LABEL_MAPPING[task]["label2id"])))
        plot_confusion_matrix(eval_golds, eval_preds,
                              classes=LABEL_MAPPING[task]["label2id"],
                              labels=label_index_to_print,
                              output_dir=output_dir,
                              title=f'Normalized confusion matrix of {task.upper()}',
                              normalize=True)
    else:
        reports: dict = classification_report(eval_golds, eval_preds,
                                              output_dict=True,
                                              zero_division=0)
        epoch_avg_f1 = reports['macro avg']['f1-score']
        epoch_avg_acc = reports['accuracy']
        LOGGER.info(f"\t{'*' * 20}Validate Summary{'*' * 20}")
        LOGGER.info(f"\tValidation Loss: {epoch_loss:.4f};\n"
                    f"\tBIO-Accuracy: {epoch_avg_acc:.4f};\n"
                    f"\tBIO-Macro-F1 score: {epoch_avg_f1:.4f}; "
                    f"\tSpend time: {datetime.timedelta(seconds=(time.time() - start_time))}")
        return epoch_loss, epoch_avg_acc, epoch_avg_f1


def train_one_epoch(model, iterator, optim, cur_epoch: int, max_grad_norm: float = 1.0, scheduler=None):
    start_time = time.time()
    tr_loss = 0.0
    model.train()
    tqdm_bar = tqdm(enumerate(iterator), total=len(iterator), desc=f'[TRAIN-EPOCH {cur_epoch}]')
    for idx, batch in tqdm_bar:
        outputs = model(**batch)
        # backward pass
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optim.zero_grad()
        outputs.loss.backward()
        optim.step()
        if scheduler:
            scheduler.step()
        tr_loss += outputs.loss.detach().item()
    epoch_loss = tr_loss / len(iterator)
    LOGGER.info(f"\t{'*' * 20}Train Summary{'*' * 20}")
    LOGGER.info(f"\tTraining Lr: {optim.param_groups[0]['lr']}; "
                f"Loss: {epoch_loss:.4f}; "
                f"Spend time: {datetime.timedelta(seconds=(time.time() - start_time))}")
    return epoch_loss


def test():
    args = get_test_argument()
    LOGGER.info(f"Arguments: {args}")
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    assert os.path.exists(args.model_path), f'Checkpoint file `{args.model_path}` not exists!'
    if device == 'cpu':
        checkpoint_data = torch.load(args.model_path, map_location='cpu')
    else:
        checkpoint_data = torch.load(args.model_path)
    configs = checkpoint_data['args']
    use_crf = True if 'crf' in args.model_arch else False
    tokenizer = AutoTokenizer.from_pretrained(configs.model_name_or_path)
    model_clss = MODEL_MAPPING[configs.model_name_or_path][configs.model_arch]
    config = AutoConfig.from_pretrained(configs.model_name_or_path,
                                        num_labels=len(checkpoint_data['classes']),
                                        finetuning_task=configs.task)
    model = model_clss(config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    LOGGER.info("Load model trained weights")
    model.load_state_dict(checkpoint_data['model'])

    test_dataset = build_dataset(args.data_dir,
                                 tokenizer,
                                 label2id=configs.label2id,
                                 header=LABEL_MAPPING[configs.task]['header'],
                                 dtype='test',
                                 max_seq_len=configs.max_seq_length,
                                 device=device,
                                 use_crf=use_crf,
                                 overwrite_data=args.overwrite_data)
    test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    # Summary
    total_params, _ = get_total_model_parameters(model)
    LOGGER.info(f"{'=' * 20}TESTER SUMMARY{'=' * 20}")
    summary_table = PrettyTable(["Parameters", "Values"])
    summary_table.add_rows([['Task', configs.task],
                            ['Model architecture', configs.model_arch],
                            ['Encoder name', configs.model_name_or_path],
                            ['Total params', total_params],
                            ['Model path', args.model_path],
                            ['Data dir', args.data_dir],
                            ['Number of examples', len(test_dataset)],
                            ['Max sequence length', configs.max_seq_length],
                            ['Test batch size', args.batch_size],
                            ['Number of workers', args.num_worker],
                            ['Use Cuda', not args.no_cuda],
                            ['Ovewrite dataset', args.overwrite_data]])
    LOGGER.info(summary_table)

    _, _, eval_f1 = validate(model=model,
                             task=args.task,
                             iterator=test_iterator,
                             cur_epoch=0,
                             is_test=True,
                             output_dir='./'
                             )


def train():
    args = get_train_argument()
    LOGGER.info(f"Arguments: {args}")
    set_ramdom_seed(args.seed)
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    use_crf = True if 'crf' in args.model_arch else False
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tensorboard_writer = SummaryWriter()
    assert os.path.isdir(args.data_dir), f'{args.data_dir} not found!'
    setattr(args, 'label2id', LABEL_MAPPING[args.task]['label2id'])
    setattr(args, 'id2label', LABEL_MAPPING[args.task]['id2label'])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset = build_dataset(args.data_dir,
                                  tokenizer,
                                  label2id=args.label2id,
                                  header=LABEL_MAPPING[args.task]['header'],
                                  dtype='train',
                                  max_seq_len=args.max_seq_length,
                                  device=device,
                                  use_crf=use_crf,
                                  overwrite_data=args.overwrite_data)
    eval_dataset = build_dataset(args.data_dir,
                                 tokenizer,
                                 label2id=args.label2id,
                                 header=LABEL_MAPPING[args.task]['header'],
                                 dtype='test',
                                 max_seq_len=args.max_seq_length,
                                 device=device,
                                 use_crf=use_crf,
                                 overwrite_data=args.overwrite_data)

    config = AutoConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=len(args.label2id),
                                        finetuning_task=args.task)
    model_clss = MODEL_MAPPING[args.model_name_or_path][args.model_arch]
    model = model_clss.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path,
                                       config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if args.load_weights is not None:
        LOGGER.info(f'Load pretrained model weights from "{args.load_weights}"')
        if device == 'cpu':
            checkpoint_data = torch.load(args.load_weights, map_location='cpu')
        else:
            checkpoint_data = torch.load(args.load_weights)
        model.load_state_dict(checkpoint_data['model'])
        checkpoint_data = None

    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    if args.model_name_or_path == 'vinai/phobert-base':
        bert_param_optimizer = list(model.roberta.named_parameters())
    else:
        bert_param_optimizer = list(model.bert.named_parameters())
    ner_param_optimizer = list(model.classifier.named_parameters())
    if 'lstm' in args.model_arch:
        ner_param_optimizer.extend(list(model.lstm.named_parameters()))
    if 'crf' in args.model_arch:
        ner_param_optimizer.extend(list(model.crf.named_parameters()))

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in ner_param_optimizer],
         'lr': args.classifier_learning_rate, 'weight_decay': args.weight_decay}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    train_steps_per_epoch = len(train_dataset) // args.train_batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=args.epochs * train_steps_per_epoch)
    train_sampler = RandomSampler(train_dataset)
    train_iterator = DataLoader(train_dataset,
                                sampler=train_sampler,
                                batch_size=args.train_batch_size,
                                num_workers=args.num_workers)
    eval_iterator = DataLoader(eval_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers)
    best_score = 0.0
    best_loss = float('inf')
    cumulative_early_steps = 0
    for epoch in range(int(args.epochs)):
        if cumulative_early_steps > args.early_stop:
            LOGGER.info(f"Early stopping. Check your saved model.")
            break
        LOGGER.info(f"\n{'=' * 30}Training epoch {epoch}{'=' * 30}")
        # Fit model with dataset
        tr_loss = train_one_epoch(model=model,
                                  optim=optimizer,
                                  iterator=train_iterator,
                                  cur_epoch=epoch,
                                  max_grad_norm=args.max_grad_norm,
                                  scheduler=scheduler)

        tensorboard_writer.add_scalar('TRAIN/Loss', tr_loss, epoch)

        # Validate trained model on dataset
        eval_loss, eval_acc, eval_f1 = validate(model=model,
                                 task=args.task,
                                 iterator=eval_iterator,
                                 cur_epoch=epoch,
                                 is_test=False)

        tensorboard_writer.add_scalar('EVAL_RESULT/Loss', eval_loss, epoch)
        tensorboard_writer.add_scalar('EVAL_RESULT/BIO-Accuracy', eval_acc, epoch)
        tensorboard_writer.add_scalar('EVAL_RESULT/BIO-F1-score', eval_f1, epoch)

        LOGGER.info(f"\t{'*' * 20}Epoch Summary{'*' * 20}")
        LOGGER.info(f"\tEpoch Loss = {eval_loss:.6f} ; Best loss = {best_loss:.6f}")
        LOGGER.info(f"\tEpoch BIO-F1 score = {eval_f1:.6f} ; Best score = {best_score:.6f}")

        if eval_loss < best_loss:
            best_loss = eval_loss
        if eval_f1 > best_score:
            cumulative_early_steps = 0
            best_score = eval_f1
            saved_file = Path(args.output_dir + f"/best_model.pt")
            LOGGER.info(f"\t***New best model, saving to {saved_file}...***")
            save_model(args, saved_file, model)
        else:
            cumulative_early_steps += 1
    if args.run_test:
        test_dataset = build_dataset(args.data_dir,
                                     tokenizer,
                                     label2id=args.label2id,
                                     header=LABEL_MAPPING[args.task]['header'],
                                     dtype='test',
                                     max_seq_len=args.max_seq_length,
                                     device=device,
                                     use_crf=use_crf,
                                     overwrite_data=args.overwrite_data)
        test_iterator = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers)
        if device == 'cpu':
            checkpoint_data = torch.load(args.output_dir + f"/best_model.pt", map_location='cpu')
        else:
            checkpoint_data = torch.load(args.output_dir + f"/best_model.pt")
        model.load_state_dict(checkpoint_data['model'])
        validate(model=model,
                 task=args.task,
                 iterator=test_iterator,
                 cur_epoch=0,
                 is_test=True,
                 output_dir=args.output_dir)


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        LOGGER.info("Start TRAIN process...")
        train()
    elif sys.argv[1] == 'test':
        LOGGER.info("Start TEST process...")
        test()
    else:
        LOGGER.error(f'[ERROR] - `{sys.argv[1]}` not found!!!')
