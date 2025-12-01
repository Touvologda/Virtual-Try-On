import streamlit as st
from streamlit import session_state as ss

def upload_person():
    st.title('Загрузка фотографии человека')
    st.write('Для наилучшего результата советуем загрузить фотографию в **полный рост**, соблюдая соотношение сторон **3:4**.')
    st.write('Если вы загрузите изображение с другими пропорциями, оно будет автоматически обрезано по центру.')
    st.write('')
    person_col, upload_col = st.columns([0.5, 0.5])

    with upload_col:
        img = st.file_uploader(
            'Загрузите фотографию человека, для которого требуется выполнить виртуальную примерку одежды.', 
            accept_multiple_files=False, 
            type=['png', 'jpeg', 'jpg'])
        if img:
            ss['person_img'] = img
            
    #initial var for starage uoload image
    if ss['person_img'] is not None:        
        with person_col:
            st.image(ss['person_img'], width=500)
            img = ss['person_img']

#set page's parametrs    
st.set_page_config(
    page_title='Upload person photo',
    page_icon=':camera_flash:',
    layout='wide',
    initial_sidebar_state='collapsed',
)

#----- main block------------
if __name__ == "__main__":
    upload_person()