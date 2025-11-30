
import logging
import os
import copy
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np

from dataset_multiTask import collate_fn_flant5
from utils import write_json, compute_metrics

logger = logging.getLogger(__name__)



def train(args, train_dataset, model, eval_dataset):
    '''Train the model'''
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset ,sampler=train_sampler, batch_size=args.BATCH_SIZE, num_workers=0, pin_memory=True, collate_fn=collate_fn_flant5)
    
    train_number=train_dataset.number
    t_total = int( (train_number / args.BATCH_SIZE * args.EPOCHS)/ args.gradient_accumulation_steps)
    warmup_steps = int(t_total * args.warmup_proportion)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer_t5 = AdamW(optimizer_grouped_parameters,
                            lr=args.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer_t5, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_number)
    logger.info("  Num Epochs = %d", args.EPOCHS)
    logger.info("  Instantaneous batch size = %d", args.BATCH_SIZE)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    best_acc, best_f1 = 0.0, 0.0
    best_epoch = 0
    best_model = None
    best_epoch=-1
    model.zero_grad()
    # results = evaluate(args, eval_dataset, model)
    train_iterator = trange(int(args.EPOCHS), desc="Epoch")
    for epoch in train_iterator:
        ra_losses, sra_losses, a_losses = 0, 0, 0
        for step,data in enumerate(tqdm(train_dataloader,desc="Iteration")):
            model.train()
            inputs, _,_ = get_input_from_batch(args, data)
            inputs['is_eval'] = False
            a_loss, sra_loss, ra_loss = model(**inputs)
            sra_losses += sra_loss.item()
            ra_losses += ra_loss.item()
            a_losses += a_loss.item()
            loss = (1-args.lamda) / 2 * sra_loss + (1-args.lamda) / 2 * ra_loss +  args.lamda * a_loss   
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer_t5.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            tr_loss += loss.item()

            # Log metrics
            if (epoch > 2 and args.logging_steps > 0 and global_step % args.logging_steps == 0) or global_step==t_total:
                results = evaluate(args, eval_dataset, model)
                all_eval_results.append(results)
                logger.info("Traing Loss: {}".format((tr_loss - logging_loss) / args.logging_steps))
                logging_loss = tr_loss

                if results['acc'] >= best_acc:
                    best_acc = results['acc']
                    best_f1 = results['f1']
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch + 1
        logger.info("ra_loss: {}, sra_loss: {}, a_loss: {}".format(ra_losses, sra_losses, a_losses))

    # Save model checkpoint
    save_model(args.output_model_file, best_model)
    readme_path = os.path.join(args.output_dir, 'readme.txt')
    with open(readme_path, 'a+') as writer:
        writer.write('Save best model at {} epoch, best_acc={}, best_f1={}'.format(best_epoch, best_acc, best_f1))
        writer.write('\n')
        
    return global_step, tr_loss/global_step, all_eval_results, best_model

def evaluate(args, eval_dataset, model, is_test=False):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.BATCH_SIZE, collate_fn=collate_fn_flant5)
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.BATCH_SIZE)

    a_pred_sequences, sra_pred_sequences, ra_pred_sequences = [], [], []
    out_label_ids = []
    rel_label_ids = []
    a_results, sra_results, ra_results = {}, {}, {}
    for batch in tqdm(eval_dataloader):
        model.eval()
        with torch.no_grad():
            inputs, relation_label,senti_labels = get_input_from_batch(args, batch)
            inputs['is_eval'] = True
            a_sequence, sra_sequence, ra_sequence = model(**inputs)

            a_pred_sequences.extend(a_sequence)
            sra_pred_sequences.extend(sra_sequence)
            ra_pred_sequences.extend(ra_sequence)
            rel_label_ids.extend(relation_label.detach().cpu().numpy())
            out_label_ids.extend(senti_labels.detach().cpu().numpy())
    
    a_preds = parse_sequences(a_pred_sequences)
    sra_preds = parse_sequences(sra_pred_sequences)
    # 
    ra_preds = parse_rel_sequences(ra_pred_sequences)

    a_result = compute_metrics(a_preds, out_label_ids)
    sra_result = compute_metrics(sra_preds, out_label_ids)
    # 
    iea_result = compute_metrics(ra_preds, rel_label_ids)
    
    a_results.update(a_result)
    sra_results.update(sra_result)
    ra_results.update(iea_result)

    results = {'a_results': a_results, 'sra_results': sra_result, 'ra_results': sra_results}
    results['avg_results'] = {}
    results['avg_results']['acc'] = results['a_results']['acc']
    results['avg_results']['f1'] = results['a_results']['f1']


    # if is_test:
    #     pred_data = []
    #     for a_p, a_s, ea_p, ea_s, iea_p, iea_s, l in zip(a_preds.tolist(), a_pred_sequences, ea_preds.tolist(), ea_pred_sequences, iea_preds.tolist(), iea_pred_sequences, out_label_ids):
    #         data = {}
    #         data['a_pred'] = a_p
    #         data['ea_pred'] = ea_p
    #         data['iea_pred'] = iea_p
    #         data['label'] = int(l)
    #         data['a_sequence'] = a_s
    #         data['ea_sequence'] = ea_s
    #         data['iea_sequence'] = iea_s
    #         pred_data.append(data)

    #     pred_file = os.path.join(args.save_model_dir, 'pred_results.json')
    #     write_json(pred_file, pred_data)
    
    return results['avg_results']


def get_input_from_batch(args, batch):
    inputs = {'a_input_ids': batch[0].to(args.device),
            'a_attention_mask': batch[1].to(args.device),
            'a_decoder_output_labels': batch[2].to(args.device),
            'sra_input_ids': batch[3].to(args.device),
            'sra_attention_mask': batch[4].to(args.device),
            'sra_decoder_output_labels': batch[5].to(args.device),
            'ra_input_ids': batch[6].to(args.device),
            'ra_attention_mask': batch[7].to(args.device),
            'ra_decoder_output_labels': batch[8].to(args.device),
            'image_feature': batch[9].to(args.device),
            }
    relation_label = batch[10].to(args.device)
    sentiment_labels = batch[11].to(args.device)
    return inputs,relation_label, sentiment_labels




def save_model(save_dir, model):
     # Save model checkpoint
    torch.save(model.state_dict(), save_dir)
    logger.info('Save best model in {}'.format(save_dir))


def parse_rel_sequences(pred_sequences):
    preds = []
    for seq in pred_sequences:
        seq = seq.lower().replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
        seq = seq.split('<relation>')[-1]
        if 'irrelvant' in seq:
            pred = 0
        else:
            pred = 1
        preds.append(pred)
    return np.array(preds)

def parse_sequences(pred_sequences):
    preds = []
    for seq in pred_sequences:
        seq = seq.lower().replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
        seq = seq.split('<emotion>')[-1]
        if 'negative' in seq:
            pred = 2
        elif 'positive' in seq:
            pred = 1
        else:
            pred = 0
        preds.append(pred)
    return np.array(preds)