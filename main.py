import streamlit as st
import litstudy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import seaborn as sns
import pandas as pd
import streamlit.components.v1 as components
from wordcloud import WordCloud
import os

# --- CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="LitStudy Pro - Analiză Bibliometrică",
    layout="wide",
    page_icon="📚"
)

# --- CSS CUSTOM ---
st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    h1 { color: #2c3e50; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("📚 LitStudy Pro: Analiză Bibliometrică Avansată")
st.markdown("### Instrument pentru analiza automată a literaturii științifice")

# --- 1. SIDEBAR: DATA LOADING ---
st.sidebar.header("1. Sursă Date")
sursa_date = st.sidebar.radio("Metoda de import:", ("Căutare Live (DBLP)", "Fișier Local"))

# Inițializăm session_state pentru a nu pierde datele la refresh (filtrare)
if 'docs' not in st.session_state:
    st.session_state['docs'] = []

# LOGICA DE ÎNCĂRCARE
new_docs = []

if 'regenerate_word_cloud' not in st.session_state:
    st.session_state['regenerate_word_cloud'] = False
regenerate_word_cloud = st.session_state['regenerate_word_cloud'] 

if sursa_date == "Căutare Live (DBLP)":
    query = st.sidebar.text_input("Cuvinte cheie", value="Machine Learning")
    limit_docs = st.sidebar.slider("Nr. maxim articole", 100, 500, 100, step=100)
    
    if st.sidebar.button("🔍 Caută pe DBLP"):
        with st.spinner('Se descarcă datele de pe DBLP...'):
            try:
                new_docs = litstudy.search_dblp(query, limit=limit_docs)
                regenerate_word_cloud = True
                st.session_state['docs'] = new_docs
                st.success(f"Găsite: {len(new_docs)} articole.")
            except Exception as e:
                st.error(f"Eroare: {e}")

else:
    uploaded_file = st.sidebar.file_uploader("Încarcă fișier (BIB, RIS, CSV)", type=["bib", "ris", "csv"])
    if uploaded_file:
        temp_name = f"temp_{uploaded_file.name}"
        with open(temp_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            with st.spinner('Se procesează fișierul...'):
                if temp_name.endswith(".bib"):
                    new_docs = litstudy.load_bibtex(temp_name)
                elif temp_name.endswith(".ris"):
                    new_docs = litstudy.load_ris(temp_name)
                elif temp_name.endswith(".csv"):
                    new_docs = litstudy.load_csv(temp_name)
                st.session_state['docs'] = new_docs
                st.sidebar.success(f"Fișier încărcat: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Eroare fișier: {e}")

# Preluăm documentele din memorie
docs = st.session_state['docs']

# --- 2. SISTEM DE FILTRARE (CERINȚA 5) ---
filtered_docs = docs
if docs:
    st.sidebar.markdown("---")
    st.sidebar.header("2. Filtrare Rezultate")
    
    # A. Filtru Ani
    years = [d.publication_year for d in docs if d.publication_year is not None]
    if years:
        min_y, max_y = int(min(years)), int(max(years))
        def slider_change_callback():
            st.session_state['regenerate_word_cloud'] = True
        sel_years = st.sidebar.slider("📅 Interval Ani", min_y, max_y, (min_y, max_y), key='my_slider', on_change=slider_change_callback)
        filtered_docs = [d for d in filtered_docs if d.publication_year and sel_years[0] <= d.publication_year <= sel_years[1]]

    # B. Filtru Sursă (Jurnal)
    sources = list(set([d.source for d in docs if hasattr(d, 'source') and d.source]))
    if sources:
        sel_source = st.sidebar.multiselect("📖 Filtru Jurnal/Conferință", sources)
        if sel_source:
            filtered_docs = [d for d in filtered_docs if hasattr(d, 'source') and d.source in sel_source]

    st.sidebar.info(f"Se analizează: **{len(filtered_docs)}** / {len(docs)} articole")

# --- 3. INTERFAȚA PRINCIPALĂ ---
if filtered_docs:
    # Definim 4 Tab-uri pentru a acoperi toate cerințele
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Statistici", 
        "🧠 Topic Modeling (NLP)", 
        "🕸️ Rețele", 
        "📥 Export Date"
    ])

    # === TAB 1: STATISTICI EXTINSE (CERINȚA 2) ===
    with tab1:
        st.subheader("Privire de ansamblu")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Publicații pe ani**")
            fig1 = plt.figure(figsize=(8, 4))
            litstudy.plot_year_histogram(filtered_docs)
            st.pyplot(fig1, use_container_width=True)
            
        with col2:
            st.markdown("**Top Autori**")
            fig2 = plt.figure(figsize=(8, 4))
            litstudy.plot_author_histogram(filtered_docs, limit=10)
            st.pyplot(fig2, use_container_width=True)

        st.markdown("---")
        
        # Statistici Extra: Surse
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Top Locații de Publicare (Venues)**")
            try:
                fig3 = plt.figure(figsize=(8, 4))
                litstudy.plot_source_histogram(filtered_docs, limit=10)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig3, use_container_width=True)
            except:
                st.warning("Nu există date despre surse.")

        with col4:
            st.markdown("**Word Cloud (Din Titluri)**")
            # Generare rapidă WordCloud
            text = " ".join([d.title for d in filtered_docs if d.title])
            if text:
                if 'wc' not in st.session_state:
                    st.session_state['wc'] = None
                wc = st.session_state['wc']
                if regenerate_word_cloud or wc is None:
                    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
                    st.session_state['wc'] = wc
                    st.session_state['regenerate_word_cloud'] = False
                fig_wc = plt.figure(figsize=(8, 4))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(fig_wc, use_container_width=True)

    # === TAB 2: TOPIC MODELING (Implementare "Low-Level" Scikit-Learn) ===
    with tab2:
        st.subheader("Detectare Automată a Subiectelor (NMF)")
        st.markdown("""
        > Această secțiune implementează algoritmul **NMF (Non-negative Matrix Factorization)** folosind direct 
        > biblioteca *Scikit-Learn* pentru o precizie maximă și control total asupra datelor.
        """)

        if len(filtered_docs) < 10:
            st.warning("⚠️ Ai nevoie de cel puțin 10 articole pentru a genera topicuri relevante.")
        else:
            col_settings, col_viz = st.columns([1, 3])
            
            with col_settings:
                st.markdown("**Setări Model**")
                num_topics = st.slider("Număr de topicuri", 3, 10, 5)
                run_nlp = st.button("🚀 Rulează Analiza")

            with col_viz:
                if run_nlp:
                    with st.spinner("Se procesează textul și se antrenează modelul NMF..."):
                        try:
                            # 1. PREGĂTIRE DATE (Extragem textul din obiectele LitStudy)
                            # Combinăm titlul cu abstractul (dacă există) pentru fiecare articol
                            text_data = []
                            for doc in filtered_docs:
                                content = doc.title
                                if hasattr(doc, 'abstract') and doc.abstract:
                                    content += " " + doc.abstract
                                text_data.append(content)

                            # 2. VECTORIZARE (TF-IDF)
                            # Transformăm textul în numere, eliminând cuvintele comune (stop words)
                            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
                            tfidf = tfidf_vectorizer.fit_transform(text_data)
                            feature_names = tfidf_vectorizer.get_feature_names_out()

                            # 3. ANTRENARE MODEL NMF
                            nmf_model = NMF(n_components=num_topics, random_state=42, init='nndsvd')
                            nmf_model.fit(tfidf)
                            
                            st.success("Analiză finalizată cu succes (Scikit-Learn Backend)!")
                            st.markdown("### 🧩 Rezultate Identificate:")

                            # 4. EXTRAGERE ȘI AFIȘARE TOPICURI
                            for topic_idx, topic in enumerate(nmf_model.components_):
                                # Luăm top 10 cuvinte cu cea mai mare greutate în topic
                                top_indices = topic.argsort()[:-11:-1]
                                top_words = [feature_names[i] for i in top_indices]
                                
                                with st.expander(f"Topic {topic_idx + 1}: {top_words[0].upper()}..."):
                                    st.write(f"**Cuvinte cheie:** {', '.join(top_words)}")
                                    # Afișăm un mini grafic de bare pentru importanța cuvintelor (Bonus vizual)
                                    topic_df = pd.DataFrame({
                                        'Cuvânt': top_words,
                                        'Importanță': topic[top_indices]
                                    })
                                    st.bar_chart(topic_df.set_index('Cuvânt'))

                        except Exception as e:
                            st.error(f"A apărut o eroare la procesare: {e}")
                            st.info("Verifică dacă articolele selectate au abstracte disponibile.")

    # === TAB 3: REȚELE ===
    with tab3:
        st.subheader("Rețea de Co-autori")
        st.info("Această vizualizare arată grupurile de cercetători care colaborează frecvent.")
        
        try:
            net = litstudy.build_coauthor_network(filtered_docs)
            if net and len(net.nodes) > 0:
                html_file = "network.html"
                litstudy.plot_network(net, height="600px")
                
                # Hack pentru a citi fișierul generat de litstudy
                # De obicei îl salvează ca 'citation.html' sau deschide temp
                if os.path.exists("citation.html"):
                    with open("citation.html", 'r', encoding='utf-8') as f:
                        html_src = f.read()
                    components.html(html_src, height=620, scrolling=True)
                else:
                    st.warning("Rețeaua a fost generată în fundal.")
            else:
                st.warning("Nu există suficiente conexiuni pentru o rețea.")
        except Exception as e:
            st.error(f"Eroare rețea: {e}")

    # === TAB 4: DATE & EXPORT (CERINȚA 3) ===
    with tab4:
        st.subheader("Export Date")
        
        # Conversie la Pandas DataFrame
        data = []
        for d in filtered_docs:
            data.append({
                "Titlu": d.title,
                "An": d.publication_year,
                "Autori": ", ".join([a.name for a in d.authors]) if d.authors else "",
                "Sursă": d.source if hasattr(d, 'source') else ""
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("### 📥 Descarcă Raport")
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📄 Descarcă Tabel (CSV)",
                data=csv,
                file_name="litstudy_export.csv",
                mime="text/csv"
            )
        
        with col_dl2:
            st.info("💡 Pentru raportul PDF complet, faceți o captură de ecran a Tab-ului 'Dashboard Statistici' și includeți-o în documentație.")

elif not docs:
    st.info("👈 Începe prin a căuta un termen sau a încărca un fișier în meniul din stânga.")