"""Entrada de Streamlit. Ejecutar con: python run.py"""

import streamlit as st

from quixote.ui import apply_style, chat_page, how_it_works_page, observability_page

# Configuración de la página de Streamlit --
st.set_page_config(page_title="Quixote",
                   page_icon="📖",
                   layout="wide",
                   initial_sidebar_state="collapsed")
# --------------------------------------------

apply_style()

# Registrar las páginas de la aplicación --
chat = st.Page(chat_page,
               title="Chat",
               icon=":material/menu_book:",
               default=True)
observability = st.Page(observability_page,
                        title="Observabilidad",
                        icon=":material/monitoring:",
                        url_path="observabilidad")
learn = st.Page(how_it_works_page,
                title="Cómo funciona",
                icon=":material/lightbulb:",
                url_path="como-funciona")
page = st.navigation([chat, observability, learn], position="hidden")
# --------------------------------------------

# Barra de navegación personalizada --
brand, first, second, third = st.columns([2.5, 1, 1.9, 1.8], vertical_alignment="center")
with brand:
    st.markdown("**Quixote**")
with first:
    st.page_link(chat, label="Chat")
with second:
    st.page_link(observability, label="Observabilidad")
with third:
    st.page_link(learn, label="Cómo funciona")
st.divider()
# --------------------------------------------

# Ejecutar la página actual
page.run()
