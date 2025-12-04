import streamlit as st
from streamlit import session_state as ss

import os
import sys
sys.path.append(ss['path'])

from typing import List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

# подключение собственных модулей
from module.pipeline_fastfit import FastFitPipeline
from parse_utils import DWposeDetector, DensePose, SCHP, multi_ref_cloth_agnostic_mask

PERSON_SIZE = (768, 1024)

#----- procedures block------------
def center_crop_to_aspect_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    width, height = img.size
    current_ratio = width / height
    
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        new_height = height
        left = (width - new_width) // 2
        top = 0
    else:
        new_width = width
        new_height = int(width / target_ratio)
        left = 0
        top = (height - new_height) // 2
    
    return img.crop((left, top, left + new_width, top + new_height))

class FastFitDemo:
    def __init__(
        self, 
        base_model_path: str = ss['path'] + 'Models/FastFit-MR-1024', 
        util_model_path: str = ss['path'] + 'Models/Human-Toolkit',
        mixed_precision: str = 'bf16',
        device: str = None,
    ):
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dwpose_detector = DWposeDetector(
            pretrained_model_name_or_path=os.path.join(util_model_path, 'DWPose'), device='cpu')
        self.densepose_detector = DensePose(model_path=os.path.join(util_model_path, 'DensePose'), device=self.device)
        self.schp_lip_detector = SCHP(ckpt_path=os.path.join(util_model_path, 'SCHP', 'schp-lip.pth'), device=self.device)
        self.schp_atr_detector = SCHP(ckpt_path=os.path.join(util_model_path, 'SCHP', 'schp-atr.pth'), device=self.device)
        self.pipeline = FastFitPipeline(
            base_model_path=base_model_path,
            device=self.device,
            mixed_precision=mixed_precision,
            allow_tf32=True
        )

    def validate_inputs(self, person_img, upper_img, lower_img, dress_img, shoe_img, bag_img) -> Tuple[bool, str]:
        has_upper = upper_img is not None
        has_lower = lower_img is not None
        has_dress = dress_img is not None

        if not (has_dress or has_upper or has_lower or shoe_img or bag_img):
            st.write('validation_no_garment')
            return False, 'validation_no_garment'
        return True, 'validation_pass'
    
    def preprocess_person_image(self, person_img: Image.Image):
        if self.dwpose_detector is None or self.densepose_detector is None or self.schp_lip_detector is None or self.schp_atr_detector is None:
            st.write('Model not initialized')
            raise RuntimeError('Model not initialized')
            
        person_img = person_img.convert('RGB')
        person_img = center_crop_to_aspect_ratio(person_img, 3/4)
        person_img = person_img.resize(PERSON_SIZE, Image.LANCZOS)
        
        pose_img = self.dwpose_detector(person_img)
        if not isinstance(pose_img, Image.Image):
            st.write('Pose estimation failed')
            raise RuntimeError('Pose estimation failed')
        
        densepose_arr = np.array(self.densepose_detector(person_img))
        lip_arr = np.array(self.schp_lip_detector(person_img))
        atr_arr = np.array(self.schp_atr_detector(person_img))
        
        return pose_img, densepose_arr, lip_arr, atr_arr
    
    def generate_mask(self, densepose_arr: np.ndarray, lip_arr: np.ndarray, atr_arr: np.ndarray, 
                     square_cloth_mask: bool = False) -> Image.Image:
        return multi_ref_cloth_agnostic_mask(
            densepose_arr, lip_arr, atr_arr,
            square_cloth_mask=square_cloth_mask,
            horizon_expand=True
        )
    
    def prepare_reference_images(self, upper_img, lower_img, dress_img, shoe_img, bag_img, ref_height: int) -> Tuple[List[Image.Image], List[str], List[int]]:
        clothing_ref_size = (int(ref_height * 3 / 4), ref_height)
        accessory_ref_size = (384, 512)
        
        ref_images, ref_labels, ref_attention_masks = [], [], []
        
        categories = [
            (upper_img, 'upper'), (lower_img, 'lower'), (dress_img, 'overall'),
            (shoe_img, 'shoe'), (bag_img, 'bag')
        ]
        
        for img, label in categories:
            target_size = accessory_ref_size if label in ['shoe', 'bag'] else clothing_ref_size
            if img is not None:
                img = img.convert("RGB").resize(target_size, Image.LANCZOS)
                ref_images.append(img)
                ref_labels.append(label)
                ref_attention_masks.append(1)
            else:
                ref_images.append(Image.new("RGB", target_size, color=(0, 0, 0)))
                ref_labels.append(label)
                ref_attention_masks.append(0)
        
        return ref_images, ref_labels, ref_attention_masks
    
    def generate_image(
        self, person_img, upper_img, lower_img, dress_img, shoe_img, bag_img,
        ref_height: int, num_inference_steps: int = 50, guidance_scale: float = 2.5,
        use_square_mask: bool = False, seed: int = 42, enable_pose: bool = True
    ) -> Tuple[Optional[Image.Image], str]:
        
        try:
            is_valid, message_key = self.validate_inputs(person_img, upper_img, lower_img, dress_img, shoe_img, bag_img)
            if not is_valid:
                st.write(message_key)
                return None, message_key
            
            if self.pipeline is None:
                st.write('error_model_not_loaded')
                return None, 'error_model_not_loaded'
            
            # MODIFIED: person_img is now a PIL.Image directly, so no dictionary handling is needed.
            if person_img is None:
                st.write('error_no_valid_person_image')
                return None, 'error_no_valid_person_image'

            processed_person_img = person_img.convert('RGB')
            processed_person_img = center_crop_to_aspect_ratio(processed_person_img, 3/4)
            processed_person_img = processed_person_img.resize(PERSON_SIZE, Image.LANCZOS)
            
            # This function does its own internal processing of the person image
            pose_img, densepose_arr, lip_arr, atr_arr = self.preprocess_person_image(person_img)
            
            # MODIFIED: Since gr.Image is used, there is no user-drawn mask.
            # We always generate the mask automatically.
            mask_img = self.generate_mask(densepose_arr, lip_arr, atr_arr, use_square_mask)
            
            ref_images, ref_labels, ref_attention_masks = self.prepare_reference_images(
                upper_img, lower_img, dress_img, shoe_img, bag_img, ref_height
            )
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            with torch.no_grad():
                result = self.pipeline(
                    person=processed_person_img, mask=mask_img, ref_images=ref_images,
                    ref_labels=ref_labels, ref_attention_masks=ref_attention_masks,
                    pose=pose_img if enable_pose else None, num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, generator=generator, return_pil=True
                )
            
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], Image.Image):
                return result[0], 'success_generation'
                
            st.write('error_generation_failed')
            return None, 'error_generation_failed'
            
        except Exception as e:
            st.write('An exception occurred:')
            st.write(e)
           
            return None, f'error_exception:{e}'

@st.cache_data(show_spinner=False)
def fetch_measures():
    # do stuff
    time.sleep(5)   

def click_button(person_image, dress_img, top_img, bottom_img):
    demo_instance = FastFitDemo()
    
    shoe_image = None
    bag_image = None
    ref_size = 512
    num_steps = 30
    guidance_scale = 2.5
    use_square_mask = False
    seed = 12345
    enable_pose = True

    #with st.spinner(text="Идет примерка ....."):
    img, msg_key = demo_instance.generate_image(person_image, top_img, bottom_img, dress_img, 
                shoe_image, bag_image, ref_size, num_steps, guidance_scale,
                use_square_mask, seed, enable_pose)
   # st.success(msg_key)
    
    
    return img, msg_key
@st.cache_data
def try_on():
    st.title('Примерочная') 
    btn_generate = st.button(
        'Примерить выбранную одежду', 
        key='btn_generate', 
        help='Нажми, чтобы примерить образ', 
        type='secondary', 
        disabled=False,
        icon='🤳'
    )
    if ss['try_on_image'] is not None:    
        st.image(ss['try_on_image']) 

        
    top_img = ss['top_img']
    bottom_img = ss['bottom_img']
    dress_img = ss['dress_img']

    if btn_generate:
        #обработаем исключения
        if ss['person_img'] is None: 
            st.write('ОШИБКА! Не выбрана модель.')
            st.page_link(ss['path'] + 'person_photo.py', label="Загрузите модель", icon='📸')
            return
        person_img = Image.open(ss['person_img'])
        
        if (dress_img is None) and \
           (top_img is None) and \
           (bottom_img is None):
            st.write('ОШИБКА! Не выбрана одежда для примерки.')
            st.page_link(ss['path'] + 'clothes.py', label="Загрузите одежду", icon='👘')
            return

        #выбрано платье
        if (ss['is_dresses']==0) and (dress_img is not None):
            dress_img = Image.open(dress_img)
            top_img = None
            bottom_img = None
        #платье может быть загружено, но выбраны отдельные изделия
        else:
            #если загружено хотя бы  одно изделие и активен выбор двух изделий, то платье не считаем
            if (top_img is not None) or (bottom_img is not None):
                dress_img = None   
            if (top_img is not None): top_img = Image.open(top_img)
            if (bottom_img is not None): bottom_img = Image.open(bottom_img)
    
        #запустим примерку    
        try_on_img, _msg_key = click_button(person_img, dress_img, \
                                      top_img, bottom_img)
        if try_on_img is not None:    
            st.image(try_on_img) 
            ss['try_on_image'] = try_on_img


#----- main block------------
if __name__ == "__main__":
    try_on()
