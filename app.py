# app.py
import streamlit as st
from streamlit import session_state as ss
import gdown
import os

#set page's parametrs    
st.set_page_config(
    page_title='Upload person photo',
    page_icon=':camera_flash:',
    layout='wide',
    #initial_sidebar_state='collapsed',
)

if 'person_img' not in ss:
    ss['person_img'] = None
    ss['path'] = ''
    ss['dress_img'] = None
    ss['top_img'] = None
    ss['bottom_img'] = None
    ss['is_dresses'] = 0
    ss['try_on_image'] = None
    ss['model_path'] = 'Models'

# Page Navigation
pages = [
    st.Page(ss['path'] + 'intro.py', title='Описание проекта', icon='👋'),
    st.Page(ss['path'] + 'person_photo.py', title='Фотография модели', icon='📸'),
    st.Page(ss['path'] + 'clothes.py', title='Образцы одежды', icon='👘'),
    st.Page(ss['path'] + 'try_on.py', title='Примерочная', icon='💃'),
   # st.Page(ss['path'] + 'switch_page_demo.py', title='st.switch_page', icon='🔀'),
]

#set name for page
st.set_page_config(page_title = 'virual try on')

# Adding pages to the sidebar navigation
pg = st.navigation(pages, position='sidebar', expanded=True)

# Running the app
pg.run()

if ss['path'] == '' :
    ss['path'] = os.getcwd() + '/'
os.chdir(ss['path'])
#st.write(not os.path.exists(ss['path']+ss['model_path']))
#if not os.path.exists(ss['path']+ss['model_path']):
#    os.makedirs('Models', exist_ok=True) 
#    url = 'https://drive.google.com/drive/folders/1v_GL73hGISRrDIM_5ig1_yCbKH9ar2I2?usp=sharing'
#    gdown.download_folder(url)
#    st.write('upload weights')

import huggingface_hub    
huggingface_hub.hf_hub_download(
    repo_id='yzd-v/DWPose',
    filename='yolox_l.onnx',
    local_dir='./Models/Human-Toolkit/DWPose/yolox_l.onnx"'
)
