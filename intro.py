import streamlit as st
from streamlit import session_state as ss

def intro():

    st.title('Виртуальная примерочная')
    st.subheader('на основе кэшируемых диффузионных моделей')
    st.image(
        ss['path'] + "log.jpeg",
        width=400
    )
    st.write("Сюда буду добавлять информацию о лицензих и краткое описание принципов работы.")

#----- main block------------
if __name__ == "__main__":
    intro()