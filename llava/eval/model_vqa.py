import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()

def set_debugger():
    from IPython.core import ultratb
    import sys
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

set_debugger()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    set_seed(0)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f,indent=4)
    f.close()

def test_single_data(args,vqa_model, tokenizer, image_processor, img_dir):
    img_path_list = sorted([item for item in os.listdir(img_dir) if not item.startswith('thumb') and item.endswith('png') ])
    img_path = []
    for idx, cur_path in enumerate(img_path_list):
        cur_img = f'slice_{str(idx)}.png'
        assert cur_img in img_path_list
        img_path.append(os.path.join(img_dir, cur_img))

    # img_path = sorted([os.path.join(img_dir, item) for item in img_path_list if not item.startswith('thumb') and item.endswith('png')])

    start_frm = int(len(img_path) * 0.33)
    end_frm = int(len(img_path) * 0.67)

    prompt1 = """
        [INST] ###Human: Suppose you are an expert in detecting Neonatal Brain Injury for Hypoxic Ischemic Encephalopathy, 
        and you are allowed to use any necessary information on the Internet for you to answer questions. 
        For now I am giving you a set of MRI scaning slices of neonatal brains, 
        these slices are marked with coressponding slice labels, like 'Slice 10' and 'Slice 11'. 
        The label means the slice depth of this slice, 
        for example, 'Slice 11' is in the middle layer between 'Slice 10' and 'Slice 12'. 
        You need to answer questions in the order they are given, and output in the predefined rules.
        Rules:
        1. For [Lesion Existence] questions, you need to decide the existence of the leison, and answer with 'yes', Or 'no'.
        2. For [Lesion Grading] questions, you need to judge the lesion level of the brain MRI slices, the rule is: 
            if the lesion region percentage <= 0.01, answer with 'level1',
            if 0.01< lesion region percentage <=0.05, answer with 'level2',
            if 0.05< lesion region percentage <=0.5, answer with 'level3',
            if 0.5< lesion region percentage <=1.0, answer with 'level4'.
        3. For [Scanner Type] questions, you need to decide the given MRI slice is scanned by GE 1.5T or SIEMENS 3T, and answer with '1.5T' or '3T'
        
    """

    question1_prompt = """
        <image>\n Yes or No: [Lesion Existence] Does a lesion exist in this brain? .
        [/INST]
    """

    question2_prompt = """
        <image>\n level1, level2, level3 or level4: [Lesion Grading] What is the severity level of brain injury in this ADC?
        [/INST]
    """
    question3_prompt = """
        <image>\n GE 1.5T or SIEMENS 3T: [Scanner Type] What is the Scanner Type of this ADC?
        [/INST]
    """
    question_list = [question1_prompt, question2_prompt, question3_prompt]
    answer_dict = {}
    for cur_idx, cur_ques in enumerate(question_list):
        img_list = []
        prompt_list = []

        real_img_path = img_path[start_frm: end_frm]
        for idx, cur_img_path in enumerate(real_img_path):
            # cur_img_path = img_path[9]
            raw_image = Image.open(cur_img_path).convert("RGB")
            img_list.append(raw_image)
        # image_tensor = process_images(img_list, image_processor, vqa_model.config)[0]
        image_tensor = process_images(img_list, image_processor, vqa_model.config)

        prompt_list = [prompt1] + [cur_ques]
        prompt = ''.join(''.join(prompt_list).split('\n'))
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            # import pdb; pdb.set_trace()
            output_ids = vqa_model.generate(
                input_ids,
                # images=image_tensor.unsqueeze(0).half().cuda(),
                images=image_tensor.half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                # use_cache=True,
                use_cache=False
                )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_name = f'ans{cur_idx}'
        answer_dict[ans_name] = outputs
    return answer_dict



def eval_model_HIEVQA(args):
    set_seed(0)
    # Model
    disable_torch_init()
    
    dataset_path = args.image_folder
    mgh_dir = os.path.join(dataset_path, 'MGH')
    dataset_split = ['BONBID2023_Train', 'BONBID2023_Test']
    for data_split in dataset_split:
        cur_split_data_dir = os.path.join(mgh_dir, data_split, '1ADC_ss')
        img_dirs = sorted(os.listdir(cur_split_data_dir))
        cur_split_answer = {}
        cur_answer_file = os.path.join(
            args.answers_file, 
            f'{data_split}_v1.json')
        
        for img_dir_idx, img_dir in enumerate(tqdm(img_dirs)):
            if not img_dir.startswith('MGHNICU'):
                continue
            model_path = os.path.expanduser(args.model_path)
            model_name = get_model_name_from_path(model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

            answer = test_single_data(
                args=args,
                vqa_model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                img_dir=os.path.join(cur_split_data_dir, img_dir)
            )
            data_id = img_dir.split('/')[-1].split('-')[0]
            cur_split_answer[data_id] = answer
            if img_dir_idx % 2 == 0:
                jsondump(cur_answer_file, cur_split_answer)
            
            del model
            torch.cuda.empty_cache()
            
        jsondump(cur_answer_file, cur_split_answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    # eval_model(args)
    eval_model_HIEVQA(args)
