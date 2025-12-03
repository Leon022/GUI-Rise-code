import os
import re
import ast
import sys
import pdb
import json
import torch
import wandb
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from accelerate.utils import gather_object
from data.data_utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda
from utils.utils import save_json
from main.utils_aitw import pred2json, pred2json_post, action2json, check_actions_match, is_tap_action

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# to avoid error;
# NUM_HISTORY = args.num_history
NUM_HISTORY = 4

import logging
logging.basicConfig(level=logging.INFO)

def broadcast_value(value, src=0, local_rank=0):
    tensor = torch.tensor([value], dtype=torch.float32).to(f'cuda:{local_rank}')
    dist.broadcast(tensor, src=src)
    return tensor.item()

def calculate_aitw_metrics(results):
    corr_action=0
    corr_type=0
    num_text=0
    corr_text=0
    num_scroll=0
    corr_scroll=0
    num_click=0
    corr_click=0
    num_both_click=0
    corr_both_click=0
    num_wrong_format=0
    num=0
    for episode in results.keys():
        for step in results[episode]:
            corr_action += step['corr_action']
            corr_type += step['corr_type']
            num_text += step['num_text']
            corr_text += step['corr_text']
            num_scroll += step['num_scroll']
            corr_scroll += step['corr_scroll']
            num_click += step['num_click']
            corr_click += step['corr_click']
            num_both_click += step['num_both_click']
            corr_both_click += step['corr_both_click']
            num_wrong_format += step['num_wrong_format']
            num += 1
    
    logging.info("[Score]: " + str(corr_action/num))
    logging.info("[Valid]: " + str(num_wrong_format/num))
    metrics = {
        "Score": corr_action / num,
        "Num Corr Action": corr_action,
        "Num Corr Type": corr_type,
        
        "Num Text": num_text,
        "Num Corr Text": corr_text,
        
        "Num Scroll": num_scroll,
        "Num Corr Scroll": corr_scroll,

        "Num Click": num_click,
        "Num Corr Click": corr_click,

        "Num Both Click": num_both_click,
        "Num Corr Both Click": corr_both_click,

        "Num Wrong Format": num_wrong_format,
        "Num": num,
    }
    return metrics

def repeat_tensor_for_batch(x, repeat_n):
    if isinstance(x, torch.Tensor):
        return x.expand(repeat_n, *x.shape[1:]).contiguous()
    elif isinstance(x, list):
        assert len(x) == 1
        return x * repeat_n
    else:
        raise TypeError(f"Unsupported type {type(x)}")

@torch.no_grad()
def validate_aitw(val_loader, model_engine, processor, epoch, global_step, writer, args):
    model_engine.eval()

    answers_unique = []
    generated_texts_unique = []
    outputs_unique = []

    global_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    metric = 0
    for input_dict in tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict, device=f'cuda:{local_rank}')
        if args.precision == "fp16":
            input_dict["pixel_values"] = input_dict["pixel_values"].half()
        elif args.precision == "bf16":
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
        else:
            input_dict["pixel_values"] = input_dict["pixel_values"].float()

        with torch.no_grad():
            forward_dict = dict(
                pixel_values=input_dict["pixel_values"],
                input_ids=input_dict["input_ids"],
                labels=input_dict["labels"],
                )
            forward_dict.update(image_grid_thw=input_dict["image_sizes"])
            forward_dict.update(patch_assign=input_dict["patch_assign"]) if "patch_assign" in input_dict else None
            forward_dict.update(patch_assign_len=input_dict["patch_assign_len"]) if "patch_assign_len" in input_dict else None
            forward_dict.update(patch_pos=input_dict["patch_pos"]) if "patch_pos" in input_dict else None
            forward_dict.update(select_mask=input_dict["select_mask"]) if "select_mask" in input_dict else None

            generate_ids = model_engine.generate(**forward_dict, 
                                    max_new_tokens=1024, 
                                    eos_token_id=processor.tokenizer.eos_token_id,
                                    use_cache=True)

            generate_ids = generate_ids[:, input_dict['input_ids'].shape[1]:]
            generated_texts = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            meta = input_dict['meta_data'][0]
            outputs = {"domain": meta['domain'],
                "anno_id": meta['anno_id'],
                "ep_id": meta['ep_id'], "img_path": meta['img_url'], "instruction": meta['task'], "sentence": generated_texts,
                "meta": meta, "annot_position": meta['annot_position']
                }
            outputs.update(dict(
                corr_action=0,
                corr_type=0,
                num_text=0,
                corr_text=0,
                num_scroll=0,
                corr_scroll=0,
                num_click=0,
                corr_click=0,
                num_both_click=0,
                corr_both_click=0,
                num_wrong_format=0,
                # num=0
            ))

            generated_texts_unique.extend(generated_texts)
            answers_unique.append(meta['answer'])
            outputs_unique.append(outputs)

    answers_unique = gather_object(answers_unique)
    generated_texts_unique = gather_object(generated_texts_unique)
    outputs_unique = gather_object(outputs_unique)

    if global_rank == 0:
        # align the settings with SeeClick
        results = {}
        each_action_acc = {}
        for pred_i, ans_i, output_i in tqdm(zip(generated_texts_unique, answers_unique, outputs_unique)):
            domain_i = output_i['domain']
            if domain_i not in results:
                results[domain_i] = {}

            ep_id = output_i['ep_id']
            if ep_id not in results[domain_i]:
                results[domain_i][ep_id] = []
            
            # step_result = output_i.copy()
            try:
                anno_id = output_i['anno_id']
                match = re.search(r'<action>(.*?)</action>', pred_i, re.DOTALL)
                new_pred_i = match.group(1).strip() if match else pred_i.strip()
                # new_pred_i = pred_i
                # new_pred_i = ast.literal_eval(new_pred_i)
                new_pred_i = pred2json(new_pred_i)
                action_pred = pred2json_post(new_pred_i)
                step_i = output_i['meta']['step']
                action_ref = action2json(step_i)

                annot_position = np.array([output_i["annot_position"][i:i + NUM_HISTORY]    \
                                            for i in range(0, len(output_i["annot_position"]), NUM_HISTORY)])
                check_match = check_actions_match(action_pred["touch_point"], 
                                                                    action_pred["lift_point"],
                                                                    action_pred["action_type"], 
                                                                    action_ref["touch_point"],
                                                                    action_ref["lift_point"], 
                                                                    action_ref["action_type"],
                                                                    annot_position)

                # step accuracy
                if check_match == True:
                    output_i['corr_action'] += 1
                    match_label = 1
                    # logging.info("Step: " + str(j) + " right")
                else:
                    match_label = 0
                    # logging.info("Step: " + str(j) + " wrong")

                # type accuracy
                if action_pred["action_type"] == action_ref["action_type"]:
                    output_i['corr_type'] += 1
                    if action_ref["action_type"] != 3 and action_ref["action_type"] != 4:
                        if action_ref["action_type"] not in each_action_acc:
                            each_action_acc[action_ref["action_type"]] = [1]
                        else:
                            each_action_acc[action_ref["action_type"]].append(1)
                else:
                    if action_ref["action_type"] != 3 and action_ref["action_type"] != 4:
                        if action_ref["action_type"] not in each_action_acc:
                            each_action_acc[action_ref["action_type"]] = [0]
                        else:
                            each_action_acc[action_ref["action_type"]].append(0)

                # text accuracy
                if action_ref["action_type"] == 3:
                    output_i['num_text'] += 1
                    if (action_pred["typed_text"] == action_ref["typed_text"]) or (
                            action_pred["typed_text"] in action_ref["typed_text"]) or (
                            action_ref["typed_text"] in action_pred["typed_text"]):
                        output_i['corr_text'] += 1

                        if action_ref["action_type"] not in each_action_acc:
                            each_action_acc[action_ref["action_type"]] = [1]
                        else:
                            each_action_acc[action_ref["action_type"]].append(1)
                    else:
                        if action_ref["action_type"] not in each_action_acc:
                            each_action_acc[action_ref["action_type"]] = [0]
                        else:
                            each_action_acc[action_ref["action_type"]].append(0)

                if action_ref["action_type"] == 4:
                    # click accuracy
                    if is_tap_action(action_ref["touch_point"], action_ref["lift_point"]):
                        output_i['num_click'] += 1
                        if match_label:
                            output_i['corr_click'] += 1
                            if action_ref["action_type"] not in each_action_acc:
                                each_action_acc[action_ref["action_type"]] = [1]
                            else:
                                each_action_acc[action_ref["action_type"]].append(1)
                        else:
                            if action_ref["action_type"] not in each_action_acc:
                                each_action_acc[action_ref["action_type"]] = [0]
                            else:
                                each_action_acc[action_ref["action_type"]].append(0)
                    # scroll accuracy
                    else:
                        output_i['num_scroll'] += 1
                        if match_label:
                            output_i['corr_scroll'] += 1
                            if action_ref["action_type"] not in each_action_acc:
                                each_action_acc[action_ref["action_type"]] = [1]
                            else:
                                each_action_acc[action_ref["action_type"]].append(1)
                        else:
                            if action_ref["action_type"] not in each_action_acc:
                                each_action_acc[action_ref["action_type"]] = [0]
                            else:
                                each_action_acc[action_ref["action_type"]].append(0)
                    if (action_pred["action_type"] == 4) and is_tap_action(action_ref["touch_point"],
                                                                            action_ref[
                                                                                "lift_point"]) and is_tap_action(
                            action_pred["touch_point"], action_pred["lift_point"]):
                        output_i['num_both_click'] += 1
                        if match_label:
                            output_i['corr_both_click'] += 1

            except Exception as e:
                print(e)
                output_i['num_wrong_format'] += 1
                print(f"format wrong with {anno_id}'s prediction: {pred_i}")

            results[domain_i][ep_id].append(output_i)

        eval_dict = {}
        for domain in results.keys():
            logging.info("==="*10)
            logging.info(f"Domain: {domain}")
            logging.info("==="*10)
            eval_dict[domain] = calculate_aitw_metrics(results[domain])

        if not args.debug:
            for domain in eval_dict.keys():
                for key, value in eval_dict[domain].items():
                    # if isinstance(value, list):
                    if key not in ['Score']:
                        continue
                    writer.add_scalar(f"metrics/aitw/{domain}/{key}", value, epoch)
                    # wandb.log({f"metrics/aitw/{domain}/{key}": value}, step=global_step)

        metric = sum([x["Score"] for x in eval_dict.values()]) / len(eval_dict)
        logging.info("==="*10)
        logging.info(f"[Avg Score]: {metric}")
        logging.info("==="*10)
        if not args.debug:
            writer.add_scalar("metrics/aitw/Avg Score", metric, epoch)
            # wandb.log({"metrics/aitw/Avg Score": metric}, step=global_step)

        save_json(results, os.path.join(args.tmp_dir, f'aitw_epo{epoch}_tmp_dict.json'))
        save_json(eval_dict, os.path.join(args.tmp_dir, f'aitw_epo{epoch}_res_dict.json'))
        save_json(each_action_acc, os.path.join(args.tmp_dir, f'aitw_epo{epoch}_each_action_acc_dict.json'))

    metric = broadcast_value(metric, src=0, local_rank=local_rank)
    return metric