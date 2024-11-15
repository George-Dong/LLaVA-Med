IMG_DIR='/nobackup/users/zfchen/code/HIEVQA/HIE_eval/MedicalEval/VLP_web_data/HIE_VQA'
QUESTION='NO'
ANSWER_PATH='data/eval'

PYTHONPATH=. python llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path microsoft/llava-med-v1.5-mistral-7b \
    --question-file $QUESTION \
    --image-folder $IMG_DIR \
    --answers-file $ANSWER_PATH \
    --temperature 1.2
    # --temperature 0.5
    # --temperature 0.0