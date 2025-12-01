import streamlit as st
from streamlit import session_state as ss

def upload_clothes():
    #Зададим заголовок страницы
    st.title('Загрузка образцов одежды для примерки')
    st.write('')

    #Определим варианты для примерки платье/не платье
    radio_val = st.radio('**ВАРИАНТЫ ДЛЯ ПРИМЕРКИ:**', ['Платье', 'Плечевое + поясное изделия'], index=ss['is_dresses'])

    #Настроим форму, если решили примерить платье
    if radio_val=='Платье':
        dress_img = st.file_uploader(
                    'Загрузите платье/пальто/сарафан и т п.', 
                    accept_multiple_files=False, 
                    type=['png', 'jpeg', 'jpg'],
                    width=400)
        if dress_img:
            ss['dress_img'] = dress_img
                
        #initial var for starage upload image
        if ss['dress_img'] is not None:        
            st.image(ss['dress_img'], width=500)
            img = ss['dress_img']
            ss['is_dresses'] = 0
    #если примерять будут не платье
    else:
        top_col, bottom_col = st.columns([0.5,0.5])
        with top_col:
            top_img = st.file_uploader(
                        'Загрузите блузку, футболку рубашку и т п.', 
                        accept_multiple_files=False, 
                        type=['png', 'jpeg', 'jpg'])
            if top_img:
                ss['top_img'] = top_img
                ss['is_dresses'] = 1
                
        #initial var for starage upload image
        if ss['top_img'] is not None:        
            with top_col:
                st.image(ss['top_img'], width=500)
                top_img = ss['top_img']

        with bottom_col:
            bottom_img = st.file_uploader(
                        'Загрузите юбку, брюки, шорты и т п.', 
                        accept_multiple_files=False, 
                        type=['png', 'jpeg', 'jpg'])
            if bottom_img:
                ss['bottom_img'] = bottom_img
                ss['is_dresses'] = 1
                
        #initial var for starage upload image
        if ss['bottom_img'] is not None:        
            with bottom_col:
                st.image(ss['bottom_img'], width=500)
                bottom_img = ss['bottom_img']
        


#----- main block------------
if __name__ == "__main__":
    upload_clothes()