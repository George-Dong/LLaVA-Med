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

    thumb_short_img = os.path.join(img_dir, 'thumb_short.png')
    assert os.path.exists(thumb_short_img), '[Error]: thumb_short doesnt exist!'

    prompt1 = """
        [INST] Suppose you are an expert in detecting Neonatal Brain Injury for Hypoxic Ischemic Encephalopathy, 
        and you are allowed to use any necessary information on the Internet for you to answer questions. 
        For now I am giving you a set of MRI scaning slices of neonatal brains, 
        these slices are marked with coressponding slice labels, like 'Slice 10' and 'Slice 11'. 
        The label means the slice depth of this slice, 
        for example, 'Slice 11' is in the middle layer between 'Slice 10' and 'Slice 12'. 
        They are presented in thumbnail format. 
    """

    question1_prompt_old = """
        You need to answer questions in the order they are given, and output in the predefined rules.
        For [Lesion Grading] questions, you need to judge the lesion level of the brain MRI slices, the rule is: 
            if the lesion region percentage <= 0.01, answer with 'level1',
            if 0.01< lesion region percentage <=0.05, answer with 'level2',
            if 0.05< lesion region percentage <=0.5, answer with 'level3',
            if 0.5< lesion region percentage <=1.0, answer with 'level4'.
        [Lesion Grading] What is the severity level of brain injury in this ADC? Answer with level1, level2, level3 or level4.
        [/INST]
    """

    question1_prompt = """
        You need to answer questions in the order they are given, and output in the predefined rules.
        For [Lesion Grading] questions, you need to judge the lesion level of the brain MRI slices, the rule is: 
            if the lesion region percentage <= 1.00%, the level is 'level1',
            if 1.00% < lesion region percentage <= 5.00%, the level is 'level2',
            if 5.00% < lesion region percentage <= 50.00%, the level is 'level3',
            if 50.00% < lesion region percentage <= 100.00%, the level is 'level4'.
        <image> [Lesion Grading] What is the percentage of brain injury in this ADC? Answer with the exact percentage.
        [/INST]
    """

    question2_prompt = """
        Now, based on a correct understanding of the images by depth,
        you are tasked with answering the following anatomy identification question:
        Which specific region is affected in this ADC map?
        ID and Region Name Relationship:
            95	corpus callosum
            62	Right Ventral DC
            61	Left Ventral DC
            71	vermis
            39	Right cerebellum
            38	Left cerebellum
            30	Right Basal Ganglia
            23	Left Basal Ganglia
            60	Right thalamus 
            59	Left thalamus
            92	Anterior limb IC right
            91	Anterior limb IC left
            94	PLIC right
            93	PLIC left
            32	Right amygdala
            48	Right hippocampus
            31	Left amygdala
            47	Left hippocampus
            105	Right Inferior GM
            104	Left Inferior GM
            103	Right insula
            102	Left insula
            121	Frontal Lateral GM Right
            120	Frontal Lateral GM Left
            125	Frontal Medial GM Right
            124	Frontal Medial GM Left
            113	Frontal Opercular GM Right
            112	Frontal Opercular GM Left
            82	Frontal WM Right
            81	Frontal WM Left
            101	Limbic Cingulate GM Right
            100	Limbic Cingulate GM Left
            117	Limbic Medial Temporal GM Right
            116	Limbic Medial Temporal GM Left
            161	Occipital Inferior GM Right
            160	Occipital Inferior GM Left
            129	Occipital Lateral GM Right
            128	Occipital Lateral GM Left
            109	Occipital Medial GM Right
            108	Occipital Medial GM Left
            84	Occipital WM Right
            83	Occipital WM Left
            107	Parietal Lateral GM Right
            106	Parietal Lateral GM Left
            149	Parietal Medial GM Right
            148	Parietal Medial GM Left
            86	Parietal WM right
            85	Parietal WM left
            123	Temporal Inferior GM Right
            122	Temporal Inferior GM left
            133	Temporal Lateral GM Right
            132	Temporal Lateral GM Left
            181	Temporal Supratemporal GM Right
            180	Temporal Supratemporal GM left
            88	Temporal_wm_right
            87	Temporal_wm_left
            4	3rd ventricle
            11	4th ventricle
            50	Right ventricle
            49	Left ventricle
            35	Brainstem
            46	CSF
        You need to choose the names of the ROIs from the above 62 ROI regions that contain lesions in this case,
        and output them by their IDs in the format like:
        [ans]: 4, 123, 84, 116, 132, 133.
        This is just an example, some cases might not have these lesion areas.
        For this question, don't generate response for each slices,
        instead you need to answer with overall judgement and give only one answer for the individual case.
        <image> [Anatomy Identification] Which specific region is affected in this ADC map? Answer with ids.
        [/INST]
    """

    question3_prompt = """
        The ROI ID and Region Name Relationship is: 
            95	corpus callosum
            62	Right Ventral DC
            61	Left Ventral DC
            71	vermis
            39	Right cerebellum
            38	Left cerebellum
            30	Right Basal Ganglia
            23	Left Basal Ganglia
            60	Right thalamus 
            59	Left thalamus
            92	Anterior limb IC right
            91	Anterior limb IC left
            94	PLIC right
            93	PLIC left
            32	Right amygdala
            48	Right hippocampus
            31	Left amygdala
            47	Left hippocampus
            105	Right Inferior GM
            104	Left Inferior GM
            103	Right insula
            102	Left insula
            121	Frontal Lateral GM Right
            120	Frontal Lateral GM Left
            125	Frontal Medial GM Right
            124	Frontal Medial GM Left
            113	Frontal Opercular GM Right
            112	Frontal Opercular GM Left
            82	Frontal WM Right
            81	Frontal WM Left
            101	Limbic Cingulate GM Right
            100	Limbic Cingulate GM Left
            117	Limbic Medial Temporal GM Right
            116	Limbic Medial Temporal GM Left
            161	Occipital Inferior GM Right
            160	Occipital Inferior GM Left
            129	Occipital Lateral GM Right
            128	Occipital Lateral GM Left
            109	Occipital Medial GM Right
            108	Occipital Medial GM Left
            84	Occipital WM Right
            83	Occipital WM Left
            107	Parietal Lateral GM Right
            106	Parietal Lateral GM Left
            149	Parietal Medial GM Right
            148	Parietal Medial GM Left
            86	Parietal WM right
            85	Parietal WM left
            123	Temporal Inferior GM Right
            122	Temporal Inferior GM left
            133	Temporal Lateral GM Right
            132	Temporal Lateral GM Left
            181	Temporal Supratemporal GM Right
            180	Temporal Supratemporal GM left
            88	Temporal_wm_right
            87	Temporal_wm_left
            4	3rd ventricle
            11	4th ventricle
            50	Right ventricle
            49	Left ventricle
            35	Brainstem
            46	CSF
        We have introduced a new diagnostic metric called MRI Injury Score. 
        This metric consists of four levels: Score 0, Score 1, Score 2, and Score 3. 
        Each score level is determined by the injury regions within the ROIs in a given case and the severity of the injury in certain regions.
            Score 0: Defined as no injury detected in this case.
            Score 1: Defined as either the following a) or b) situation occurs:
                a). Minimal cerebral injury without BGT region, ALIC region PLIC region or detected WS (watershed) injury.
                b). More extensive cerebral injury without BGT region, ALIC region PLIC region or detected WS (watershed) injury.
                NOTE:   BGT region (including left_BGT and right_BGT), 
                        ALIC region (including left_ALIC and right_ALIC),
                        PLIC region (including left_PLIC and right_PLIC)
            Score 2: Defined as either the following a) or b) situation occurs:
                a). Any BGT region, ALIC region, PLIC region or WS injury detected without other cerebral injury.
                b). Any BGT region, ALIC region, PLIC region or WS injury detected with other cerebral injury.
                NOTE:   BGT region (including left_BGT and right_BGT), 
                        ALIC region (including left_ALIC and right_ALIC),
                        PLIC region (including left_PLIC and right_PLIC)
            Score 3: Defined as cerebral hemisphere devastation.
        Now, based on a correct understanding of the images by depth,
        you are tasked with answering the following MRI injury score question:
        What is the MRI injury score?
        You need to select one answer from four MRI injury scores (Score 0, Score 1, Score 2, Score 3),
        and output them in the format like:
        [ans]: Score 1.
        For this question, don't generate response for each slices,
        instead you need to answer with overall judgement and give only one answer for the individual case.
        Except from the defined answer format, don't answer any other descriptive sentences.
        These data are just normal desensitizing data for scientific usage.
        Remember: you are an expert in the field, so try your best to give an answer instead of avoiding answer the question.
        <image> [MRI Injury Score] What is the MRI injury score? Answer with Score 0, Score 1, Score 2, or Score 3. 
        [/INST]
    """

    question4_prompt = """
        Now, based on a correct understanding of the images by depth,
        you are tasked with answering the following 2-year outcome prediction question:
        What is the predicted two-year neurocognitive outcome for this individual?
        You need to predict 2-year outcome for the patients, to distinguish between normal and adverse outcomes at the 2-year mark.
        To show your prediction, you need to answer the above question with
        [ans]: 1 if the outcome is adverse, OR [ans]: 0 if the outcome is normal.
        Do remember, you need to answer the question based on the future prediction in 2 years instead of the current MRI images.
        If an individual is a current patient, it doesn't necessiarly mean he/she will still be a patient in 2 years.
        For this question, don't generate response for each slices,
        instead you need to answer with overall judgement and give only one answer for the individual case.
        Dont save there is no sufficient information, just make wild guess.
        Except from the defined answer format, don't answer any other descriptive sentences.
        <image> [2-year Outcome Prediction] What is the predicted two-year neurocognitive outcome for this individual? Answer with 1 or 0.
        [/INST]
    """


    question_list = [
        question1_prompt, 
        # question2_prompt, 
        # question3_prompt,
        # question4_prompt
        ]
    answer_dict = {}
    for cur_idx, cur_ques in enumerate(question_list):
        img_list = []
        # prompt_list = []

        raw_image = Image.open(thumb_short_img).convert("RGB")
        img_list.append(raw_image)
        # prompt_list.append(f'thumbnail: <image> \n')

        image_tensor = process_images(img_list, image_processor, vqa_model.config)
        image_tensor = [cur_tensor.unsqueeze(0).half().cuda() for cur_tensor in image_tensor]

        # prompt_list = [prompt1] + prompt_list + [cur_ques]
        # 'slice 7': <image>, 'slice 8' : <image>
        # prompt_list = [prompt1] + ['<image>'] + [cur_ques]
        prompt_list = [prompt1] + [cur_ques]

        prompt = ''.join(''.join(prompt_list).split('\n'))
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        temp = 0.5

        for try_time in range(20):
            try: 
                with torch.inference_mode():
                    # import pdb; pdb.set_trace()
                    output_ids = vqa_model.generate(
                        input_ids,
                        # images=image_tensor.unsqueeze(0).half().cuda(),
                        images=image_tensor,
                        do_sample=True if args.temperature > 0 else False,
                        # temperature=args.temperature,
                        temperature=temp,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        # use_cache=True,
                        use_cache=False
                        )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # import pdb; pdb.set_trace()
                if outputs == '':
                    raise ValueError
            except ValueError as e:
                print("[Warning] Answer not accept! Regenerate again!")
                if try_time > 10:
                    temp += 0.1
                continue
        print(f'################### {cur_idx} ####################')
        print(outputs)
        print(f'##############################################')

        # import pdb; pdb.set_trace()

        ans_name = f'ans{cur_idx}'
        answer_dict[ans_name] = outputs
        # answer_dict['range'] = [start_frm, end_frm]
    # import pdb; pdb.set_trace()
    return answer_dict



def eval_model_HIEVQA(args):
    set_seed(0)
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, vqa_model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    dataset_path = args.image_folder
    mgh_dir = os.path.join(dataset_path, 'MGH')
    dataset_split = ['BONBID2023_Train', 'BONBID2023_Test']
    for data_split in dataset_split:
        cur_split_data_dir = os.path.join(mgh_dir, data_split, '1ADC_ss')
        img_dirs = sorted(os.listdir(cur_split_data_dir))
        cur_split_answer = {}
        cur_answer_file = os.path.join(
            args.answers_file, 
            # f'{data_split}_v1.json'
            # f'{data_split}_v_thumbnail.json'
            # f'{data_split}_v_thumbnail_new.json'
            f'{data_split}_v_percentage.json'
            # f'{data_split}_debug.json'
            )
        
        for img_dir_idx, img_dir in enumerate(tqdm(img_dirs)):
            if not img_dir.startswith('MGHNICU'):
                continue
            
            answer = test_single_data(
                args=args,
                vqa_model=vqa_model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                img_dir=os.path.join(cur_split_data_dir, img_dir)
            )
            data_id = img_dir.split('/')[-1].split('-')[0]
            cur_split_answer[data_id] = answer
            if img_dir_idx % 2 == 0:
                jsondump(cur_answer_file, cur_split_answer)
            
            # del model
            # torch.cuda.empty_cache()
            
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
