import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
from datetime import datetime
import seaborn as sns
from wordcloud import WordCloud
import spacy
from collections import Counter
import string
import warnings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
warnings.filterwarnings('ignore')
import gdown
import zipfile
import os

# Page config
st.set_page_config(
    page_title="E-commerce Analytics Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def install_and_load_spacy_model():
    """Install and load spaCy Portuguese model with fallback"""
    try:
        import spacy
        # Try to load the model
        nlp = spacy.load("pt_core_news_sm")
        return nlp
    except OSError:
        # Model not found, try to install it
        try:
            st.info("Installing Portuguese language model for text analysis...")
            import subprocess
            import sys
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "pt_core_news_sm"
            ])
            # Try to load again after installation
            import spacy
            nlp = spacy.load("pt_core_news_sm")
            st.success("Portuguese language model installed successfully!")
            return nlp
        except Exception as e:
            st.warning("Could not install Portuguese language model. Using fallback mode.")
            st.info("Text analysis will continue with basic tokenization.")
            # Return a simple fallback object
            return create_fallback_nlp()

def create_fallback_nlp():
    """Create a fallback NLP processor when spaCy is not available"""
    class FallbackNLP:
        def __call__(self, text):
            # Simple tokenization fallback
            tokens = text.lower().split()
            return [FallbackToken(token) for token in tokens]
    
    class FallbackToken:
        def __init__(self, text):
            self.text = text
            # Simple POS tagging based on word endings (very basic)
            if text.endswith(('ão', 'o', 'a', 'e')):
                self.pos_ = "NOUN"
            elif text.endswith(('ado', 'ida', 'oso', 'osa')):
                self.pos_ = "ADJ"
            else:
                self.pos_ = "OTHER"
    
    return FallbackNLP()

# Get Portuguese stopwords
@st.cache_data
def get_portuguese_stopwords():
    """Get Portuguese stopwords with fallback"""
    try:
        nlp = spacy.load("pt_core_news_sm")
        return nlp.Defaults.stop_words
    except:
        # Fallback Portuguese stopwords
        return {
            'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'até', 'com', 'como', 
            'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'ela', 'elas', 
            'ele', 'eles', 'em', 'entre', 'era', 'eram', 'essa', 'essas', 'esse', 'esses', 'esta', 'estamos', 
            'estar', 'estas', 'estava', 'estavam', 'este', 'esteja', 'estejam', 'estejamos', 'estes', 'esteve', 
            'estive', 'estivemos', 'estiver', 'estivera', 'estiveram', 'estiverem', 'estivermos', 'estivesse', 
            'estivessem', 'estivéramos', 'estivéssemos', 'estou', 'está', 'estávamos', 'estão', 'eu', 'foi', 
            'fomos', 'for', 'fora', 'foram', 'forem', 'formos', 'fosse', 'fossem', 'fui', 'fôramos', 
            'fôssemos', 'haja', 'hajam', 'hajamos', 'havemos', 'havia', 'haviam', 'havido', 'havidos', 'haver', 
            'haveremos', 'haveria', 'haveriam', 'haveríamos', 'houve', 'houvemos', 'houver', 'houvera', 
            'houveram', 'houverei', 'houverem', 'houveremos', 'houveria', 'houveriam', 'houveríamos', 
            'houvermos', 'houvesse', 'houvessem', 'houvéramos', 'houvéssemos', 'há', 'hão', 'isso', 'isto', 
            'já', 'lhe', 'lhes', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 
            'na', 'nas', 'nem', 'no', 'nos', 'nossa', 'nossas', 'nosso', 'nossos', 'não', 'nós', 'o', 'os', 
            'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'são', 
            'se', 'seja', 'sejam', 'sejamos', 'sem', 'ser', 'sera', 'seremos', 'seria', 'seriam', 'seríamos', 
            'seu', 'seus', 'só', 'sua', 'suas', 'também', 'te', 'tem', 'temos', 'tenha', 'tenham', 'tenhamos', 
            'tenho', 'ter', 'terei', 'teremos', 'teria', 'teriam', 'teríamos', 'teu', 'teus', 'teve', 'tinha', 
            'tinham', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram', 'tiverem', 'tivermos', 'tivesse', 
            'tivessem', 'tivéramos', 'tivéssemos', 'tu', 'tua', 'tuas', 'tém', 'tínhamos', 'um', 'uma', 
            'você', 'vocês', 'vos', 'à', 'às', 'éramos'
        }

@st.cache_data
def load_data():
    """Load all datasets silently from Google Drive"""
    try:
        os.makedirs('data', exist_ok=True)

        drive_files = {
            'customers_dataset.csv': '1MbHCiu8ZbJies0NQ0sTCV4_rESKuDvnJ',
            'geolocation_dataset.csv': '1VK7B0Cm9RQmJIKljrvVirphZiTEZFdhH',
            'order_items_dataset.csv': '1TNfwU1jvMKNaLDRYpA9TDvU77gaAQDmT',
            'order_payments_dataset.csv': '14b96-g7a2rnM47Ml9axgbcyGMmziW5Md',
            'order_reviews_dataset.csv': '1hPfX7FO6jHW171FYeTN4QS39hS0K_4CX',
            'orders_dataset.csv': '1dPxq9qXTSZjrdXQk8IJE3i4U9EEveuw0',
            'product_category_name_translation.csv': '1H-loSFk7Ef4C6ikjcseb5_kGj7uZ3__l',
            'products_dataset.csv': '1N8KxKyxHtvae_Gyw6d8btBZBUMxqVRC2',
            'sellers_dataset.csv': '1DAhnXFWFLy84dgsGkCapZB5_2c9x4F_U'
        }

        for filename, file_id in drive_files.items():
            filepath = f'data/{filename}'
            if not os.path.exists(filepath):
                url = f'https://drive.google.com/uc?export=download&id={file_id}'
                gdown.download(url, filepath, quiet=True)

        required_files = list(drive_files.keys())
        for filename in required_files:
            if not os.path.exists(f'data/{filename}'):
                return None  # Exit if any file missing

        # Load CSVs
        customers = pd.read_csv('data/customers_dataset.csv')
        geolocation = pd.read_csv('data/geolocation_dataset.csv')
        order_items = pd.read_csv('data/order_items_dataset.csv')
        order_payments = pd.read_csv('data/order_payments_dataset.csv')
        order_reviews = pd.read_csv('data/order_reviews_dataset.csv')
        orders = pd.read_csv('data/orders_dataset.csv')
        product_translation = pd.read_csv('data/product_category_name_translation.csv')
        products = pd.read_csv('data/products_dataset.csv')
        sellers = pd.read_csv('data/sellers_dataset.csv')

        return orders, customers, order_items, order_payments, order_reviews, products, product_translation, geolocation, sellers

    except:
        return None

def preprocess_data(orders, customers, order_items, order_payments, order_reviews, products, product_translation, geolocation, sellers):
    """Preprocess data for analysis"""
    
    # Convert datetime columns
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'], errors='coerce')
    orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'], errors='coerce')
    
    # Calculate delivery metrics
    orders['actual_delivery_days'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
    orders['estimated_delivery_days'] = (orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']).dt.days
    orders['delivery_delay'] = orders['actual_delivery_days'] - orders['estimated_delivery_days']
    orders['is_delayed'] = orders['delivery_delay'] > 0
    
    # Clean review text
    order_reviews['review_comment_message'] = order_reviews['review_comment_message'].fillna('')
    
    return orders, customers, order_items, order_payments, order_reviews, products, product_translation, geolocation, sellers

def label_sentiment(score):
    """Label sentiment based on review score"""
    if score >= 4:
        return 'Positive'
    elif score <= 2:
        return 'Negative'
    else:
        return 'Neutral'

def clean_text(text, stop_words):
    """Clean and tokenize text"""
    if pd.isna(text) or text == '':
        return []
    
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', str(text))
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Lowercase and split
    tokens = text.lower().split()
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return tokens

def extract_noun_adj_bigrams(tokens, nlp):
    """Extract noun + adjective bigrams using spaCy or fallback"""
    if not tokens or nlp is None:
        return []

    try:
        # Fallback mode: detect manually
        if type(nlp).__name__ == "FallbackNLP":
            # Ambil semua pasangan 2 kata berturut-turut
            return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]

        # Jika spaCy beneran jalan
        text = " ".join(tokens)
        doc = nlp(text)
        bigrams = []
        for i in range(len(doc) - 1):
            token1 = doc[i]
            token2 = doc[i + 1]
            if token1.pos_ in ["NOUN", "PROPN"] and token2.pos_ in ["ADJ", "ADV"]:
                bigrams.append(f"{token1.text} {token2.text}")
        return bigrams
    except Exception as e:
        return []

def display_kpi_summary(orders, order_reviews, order_payments):
    """Display KPI summary metrics"""
    
    # Calculate metrics
    total_orders = len(orders)
    avg_review_score = order_reviews['review_score'].mean()
    avg_delivery_days = orders['actual_delivery_days'].mean()
    total_transaction_value = order_payments['payment_value'].sum()
    
    # Display in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📦 Total Orders",
            value=f"{total_orders:,}"
        )
    
    with col2:
        st.metric(
            label="⭐ Avg Review Score",
            value=f"{avg_review_score:.2f}/5"
        )
    
    with col3:
        st.metric(
            label="🚚 Avg Delivery Time",
            value=f"{avg_delivery_days:.1f} days"
        )
    
    with col4:
        st.metric(
            label="💰 Total Transaction Value",
            value=f"R$ {total_transaction_value/1000000:.1f}M"
        )

def menu_delivery_evaluation(orders, customers, geolocation):
    """Menu 1: Delivery Evaluation"""
    st.header("Menu 1: Delivery Evaluation (Logistics Performance)")

    # Merge data for analysis
    delivery_data = orders.merge(customers, on='customer_id', how='left')

    # Group by state
    state_metrics = delivery_data.groupby('customer_state').agg({
        'order_id': 'count',
        'delivery_delay': 'mean',
        'is_delayed': 'mean'
    }).reset_index()

    state_metrics.columns = ['State', 'Total_Orders', 'Avg_Delay_Days', 'Delay_Percentage']
    state_metrics['Delay_Percentage'] = state_metrics['Delay_Percentage'] * 100
    state_metrics = state_metrics.dropna()

    # Ambil rata-rata lat/lon per provinsi dari geolocation
    geo_avg = geolocation.groupby('geolocation_state').agg({
        'geolocation_lat': 'mean',
        'geolocation_lng': 'mean'
    }).reset_index()
    geo_avg.columns = ['State', 'Lat', 'Lon']

    # Merge metrics with geo
    state_metrics_geo = state_metrics.merge(geo_avg, on='State', how='left').dropna(subset=['Lat', 'Lon'])

    # Mapping kode provinsi ke nama lengkap
    state_name_map = {
        'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas', 'BA': 'Bahia',
        'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo', 'GO': 'Goiás',
        'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul', 'MG': 'Minas Gerais',
        'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná', 'PE': 'Pernambuco', 'PI': 'Piauí',
        'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte', 'RS': 'Rio Grande do Sul',
        'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina', 'SP': 'São Paulo',
        'SE': 'Sergipe', 'TO': 'Tocantins'
    }
    state_metrics_geo['State_Full'] = state_metrics_geo['State'].map(state_name_map)

    # Format angka untuk tooltip
    state_metrics_geo['Total_Orders_fmt'] = state_metrics_geo['Total_Orders'].astype(int)
    state_metrics_geo['Avg_Delay_Days_fmt'] = state_metrics_geo['Avg_Delay_Days'].round(2)
    state_metrics_geo['Delay_Percentage_fmt'] = state_metrics_geo['Delay_Percentage'].round(2)

    # Hover text
    state_metrics_geo['hover_text'] = (
        state_metrics_geo['State_Full'] + "<br>" +
        "Total orders: " + state_metrics_geo['Total_Orders_fmt'].astype(str) + "<br>" +
        "Average delivery delays: " + state_metrics_geo['Avg_Delay_Days_fmt'].astype(str) + " days<br>" +
        "Delay percentage: " + state_metrics_geo['Delay_Percentage_fmt'].astype(str) + "%"
    )

    # GeoJSON URL
    geojson_url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'

    # Choropleth map
    fig_map = px.choropleth_mapbox(
        state_metrics_geo,
        geojson=geojson_url,
        locations='State_Full',
        featureidkey='properties.name',
        color='Avg_Delay_Days',
        color_continuous_scale='Reds_r',
        range_color=[state_metrics_geo['Avg_Delay_Days'].min(), 0],
        mapbox_style='carto-positron',
        zoom=3,
        center={"lat": -14.2350, "lon": -51.9253},
        opacity=0.7,
        title='Average Delivery Delay by State',
        custom_data=['hover_text']
    )

    # Tooltip tampil sesuai format custom
    fig_map.update_traces(
        hovertemplate="%{customdata[0]}"
    )

    fig_map.update_layout(
        height=600,
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )

    # Tampilkan di Streamlit
    st.plotly_chart(fig_map, use_container_width=True)

    # Bar chart TOP 5 States dengan Average Delivery Delay TERTINGGI (paling jauh dari 0)
    top5_worst = state_metrics_geo.sort_values('Avg_Delay_Days', ascending=True).head(5)  # ascending=True karena nilai negatif, yang terkecil = terjauh dari 0
    
    fig_delay_worst = px.bar(
        top5_worst.sort_values('Avg_Delay_Days', ascending=False),  # sort descending untuk tampil Acre di atas (yang terburuk)
        x='Avg_Delay_Days',
        y='State_Full',
        orientation='h',
        color='Avg_Delay_Days',
        color_continuous_scale='Reds',
        title='Top 5 States with Highest Delivery Delay (Furthest from 0)',
        labels={
            'State_Full': 'State',
            'Avg_Delay_Days': 'Average Delivery Delay (days)'
        },
        text='Avg_Delay_Days'
    )

    fig_delay_worst.update_layout(
        height=400,
        margin=dict(l=120),
        showlegend=False
    )
    
    fig_delay_worst.update_traces(
        marker=dict(line=dict(width=0)),
        textposition='auto',
        texttemplate='%{text:.2f}',
        textfont_size=11
    )
    
    # Update colorscale agar Acre (yang terburuk) berwarna merah pekat
    fig_delay_worst.update_coloraxes(reversescale=True)
    
    st.plotly_chart(fig_delay_worst, use_container_width=True)

    # Bar chart TOP 5 States dengan Average Delivery Delay TERENDAH (paling mendekati 0)
    top5_best = state_metrics_geo.sort_values('Avg_Delay_Days', ascending=False).head(5)  # ascending=False karena nilai negatif, yang terbesar = terdekat ke 0
    
    fig_delay_best = px.bar(
        top5_best.sort_values('Avg_Delay_Days', ascending=True),  # sort ascending untuk tampil dari yang terdekat 0 di atas
        x='Avg_Delay_Days',
        y='State_Full',
        orientation='h',
        color='Avg_Delay_Days',
        color_continuous_scale='Greens_r',  # Gunakan warna hijau untuk yang terbaik
        title='Top 5 States with Lowest Delivery Delay (Closest to 0)',
        labels={
            'State_Full': 'State',
            'Avg_Delay_Days': 'Average Delivery Delay (days)'
        },
        text='Avg_Delay_Days'
    )

    fig_delay_best.update_layout(
        height=400,
        margin=dict(l=120),
        showlegend=False
    )
    
    fig_delay_best.update_traces(
        marker=dict(line=dict(width=0)),
        textposition='auto',
        texttemplate='%{text:.2f}',
        textfont_size=11
    )
    
    st.plotly_chart(fig_delay_best, use_container_width=True)

def menu_sentiment_analysis(order_reviews, order_items, products, product_translation, nlp, stop_words):
    st.header("MENU 2: Analisis Sentimen Aspek 'Pengiriman' dan 'Produk'")
    
    # Get preprocessed data from session state
    if 'df' not in st.session_state:
        st.error("Data belum diproses. Silakan refresh halaman.")
        return
    
    df = st.session_state.df
    
    # Inisialisasi VADER
    vader_analyzer = SentimentIntensityAnalyzer()
    
    # Fungsi label sentimen dengan VADER
    def label_sentiment_vader(text):
        if not text or pd.isna(text):
            return 'Neutral'
        score = vader_analyzer.polarity_scores(text)
        compound = score['compound']
        if compound >= 0.05:
            return 'Positive'
        elif compound <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    # Fungsi klasifikasi ABSA dengan VADER
    def klasifikasi_absa_vader(text):
        hasil = {
            "Buyer Preferences": None,
            "Product Quality": None,
            "Delivery Performance": None
        }
    
        aspek_dict = {
            "Buyer Preferences": ["gostei", "odiei", "amei", "detesto"],
            "Product Quality": ["produto", "qualidade", "defeito", "quebrado", "bom"],
            "Delivery Performance": ["entrega", "prazo", "transportadora", "rápido", "atrasado"]
        }
    
        for aspek, keywords in aspek_dict.items():
            if any(k in text.lower() for k in keywords):
                hasil[aspek] = label_sentiment_vader(text)
    
        return hasil
    
    # Process ABSA
    df["aspek_sentimen"] = df["review_comment_message"].apply(klasifikasi_absa_vader)
    
    # Hitung hasil
    aspect_counts = {
        "Buyer Preferences": {"Positive": 0, "Neutral": 0, "Negative": 0},
        "Product Quality": {"Positive": 0, "Neutral": 0, "Negative": 0},
        "Delivery Performance": {"Positive": 0, "Neutral": 0, "Negative": 0}
    }
    
    for hasil in df["aspek_sentimen"]:
        for aspek, label in hasil.items():
            if label in ["Positive", "Negative"]:
                aspect_counts[aspek][label] += 1
    
    # Convert ke DataFrame
    aspect_df = pd.DataFrame(aspect_counts).T
    aspect_df = aspect_df[['Positive', 'Negative']]
    aspect_df = aspect_df.reset_index().rename(columns={'index': 'Aspect'})
    
    # Stacked Bar Chart with labels
    fig_aspect_bar = go.Figure(data=[
        go.Bar(
            name='Positive',
            x=aspect_df['Positive'],
            y=aspect_df['Aspect'],
            orientation='h',  # Horizontal bar
            marker_color='#d2601a',
            text=aspect_df['Positive'],
            textposition='inside',
            insidetextanchor='start',
            textfont=dict(color='white', size=16)  # Perbesar teks
        ),
        go.Bar(
            name='Negative',
            x=aspect_df['Negative'],
            y=aspect_df['Aspect'],
            orientation='h',
            marker_color='#f4b183',
            text=aspect_df['Negative'],
            textposition='inside',
            insidetextanchor='end',
            textfont=dict(color='black', size=16)  # Perbesar teks
        )
    ])
    
    fig_aspect_bar.update_layout(
        barmode='stack',
        title='Sentiment Distribution by Aspect (Positive vs Negative)',
        xaxis_title='Count',
        yaxis_title='Aspect',
        title_font=dict(size=20),
        xaxis=dict(
            showgrid=False,  # Hapus gridline x
            title_font=dict(size=16),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            showgrid=False,  # Hapus gridline y
            title_font=dict(size=16),
            tickfont=dict(size=14)
        ),
        plot_bgcolor='white',  # Warna background bersih
        height=400
    )
    
    st.plotly_chart(fig_aspect_bar, use_container_width=True)
    
    # Bigram Analysis
    st.subheader("Bigram Analysis (Noun + Adjective)")
    
    # Collect all bigrams
    all_bigrams = []
    for bigrams in df['noun_adj_bigrams']:
        if bigrams:  # Check if not empty
            all_bigrams.extend(bigrams)
    
    if all_bigrams:
        bigram_counts = Counter(all_bigrams)
        top_bigrams = dict(bigram_counts.most_common(20))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # WordCloud
            if top_bigrams:
                wc = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='viridis'
                ).generate_from_frequencies(top_bigrams)
                
                fig_wc, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('WordCloud of Noun + Adjective Bigrams', fontsize=16)
                st.pyplot(fig_wc)
        
        with col2:
            # Top bigrams bar chart
            if len(top_bigrams) > 0:
                bigram_df = pd.DataFrame(list(top_bigrams.items()), columns=['Bigram', 'Count'])
                bigram_df = bigram_df.sort_values('Count', ascending=True).tail(15)
                
                fig_bigram = px.bar(
                    bigram_df,
                    x='Count',
                    y='Bigram',
                    orientation='h',
                    title='Top 15 Noun + Adjective Bigrams',
                    color='Count',
                    color_continuous_scale='viridis'
                )
                fig_bigram.update_layout(height=500)
                st.plotly_chart(fig_bigram, use_container_width=True)
    else:
        st.warning("No bigrams found in the review data.")

    # Product analysis
    st.subheader("Product Performance Analysis")
    
    # Merge product data
    product_reviews = order_items.merge(order_reviews, on='order_id', how='left')
    product_reviews = product_reviews.merge(products, on='product_id', how='left')
    product_reviews = product_reviews.merge(product_translation, on='product_category_name', how='left')
    
    # Hitung metrik per kategori
    category_metrics = product_reviews.groupby('product_category_name_english').agg({
        'order_id': 'count',
        'review_score': 'mean'
    }).reset_index()
    category_metrics.columns = ['Category', 'Total Sales', 'Average Score Review']
    category_metrics = category_metrics.dropna().sort_values('Total Sales', ascending=False).head(15)
    
    # Tentukan warna teks berdasarkan nilai
    threshold_sales = category_metrics['Total Sales'].median()
    threshold_review = category_metrics['Average Score Review'].median()
    
    category_metrics['Sales_Text_Color'] = category_metrics['Total Sales'].apply(
        lambda x: 'white' if x > threshold_sales else 'black'
    )
    category_metrics['Review_Text_Color'] = category_metrics['Average Score Review'].apply(
        lambda x: 'white' if x > threshold_review else 'black'
    )
    
    # Buat layout 2 kolom
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sales = px.bar(
            category_metrics.sort_values('Total Sales', ascending=True).tail(10),
            x='Total Sales',
            y='Category',
            orientation='h',
            color='Total Sales',
            color_continuous_scale='Blues',
            text='Total Sales'
        )
        fig_sales.update_traces(
            texttemplate='%{text}',
            textposition='inside',
            insidetextfont=dict(color=category_metrics.sort_values('Total Sales', ascending=True).tail(10)['Sales_Text_Color'])
        )
        fig_sales.update_layout(
            height=400,
            title='Top 10 Categories by Sales',
            coloraxis_showscale=False  # HILANGKAN COLORBAR
        )
        st.plotly_chart(fig_sales, use_container_width=True)
    
    with col2:
        fig_review = px.bar(
            category_metrics.sort_values('Average Score Review', ascending=True).tail(10),
            x='Average Score Review',
            y='Category',
            orientation='h',
            color='Average Score Review',
            color_continuous_scale='Greens',
            text='Average Score Review'
        )
        fig_review.update_traces(
            texttemplate='%{text:.2f}',
            textposition='inside',
            insidetextfont=dict(color=category_metrics.sort_values('Average Score Review', ascending=True).tail(10)['Review_Text_Color'])
        )
        fig_review.update_layout(
            height=400,
            title='Top 10 Categories by Review Score',
            coloraxis_showscale=False  # HILANGKAN COLORBAR
        )
        st.plotly_chart(fig_review, use_container_width=True)

def menu_market_expansion(orders, customers, order_reviews):
    """Menu 3: Market Expansion Opportunities"""
    st.header("Menu 3: Market Expansion Opportunities")
    
    # Merge data
    market_data = orders.merge(customers, on='customer_id', how='left')
    market_data = market_data.merge(order_reviews, on='order_id', how='left')
    
    # Group by state
    state_analysis = market_data.groupby('customer_state').agg({
        'order_id': 'count',
        'review_score': 'mean'
    }).reset_index()
    state_analysis.columns = ['State', 'Order_Count', 'Avg_Review_Score']
    state_analysis = state_analysis.dropna()
    
    # Calculate medians for quadrant analysis
    median_orders = state_analysis['Order_Count'].median()
    median_review = state_analysis['Avg_Review_Score'].median()
    
    # Classify states into quadrants
    def classify_quadrant(row):
        if row['Order_Count'] >= median_orders and row['Avg_Review_Score'] >= median_review:
            return 'Strong Market'
        elif row['Order_Count'] < median_orders and row['Avg_Review_Score'] >= median_review:
            return 'Expansion Target'
        elif row['Order_Count'] < median_orders and row['Avg_Review_Score'] < median_review:
            return 'Evaluate/Leave'
        else:
            return 'Volume Leader'
    
    state_analysis['Market_Segment'] = state_analysis.apply(classify_quadrant, axis=1)
    
    # Scatter plot
    fig_scatter = px.scatter(
        state_analysis,
        x='Avg_Review_Score',
        y='Order_Count',
        color='Market_Segment',
        size='Order_Count',
        hover_data=['State'],
        title='Market Expansion Opportunities'
    )
    
    # Add quadrant lines
    fig_scatter.add_hline(y=median_orders, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.add_vline(x=median_review, line_dash="dash", line_color="gray", opacity=0.5)
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Show expansion targets
    expansion_targets = state_analysis[state_analysis['Market_Segment'] == 'Expansion Target']
    if len(expansion_targets) > 0:
        st.subheader("Expansion Targets")
        st.dataframe(expansion_targets.sort_values('Avg_Review_Score', ascending=False))

def main():
    st.title("E-commerce Analytics Dashboard")
    st.markdown("### Comprehensive Analysis for E-commerce Business Optimization")
    
    # Load spaCy model and stopwords
    nlp = install_and_load_spacy_model()  # Fixed function name
    stop_words = get_portuguese_stopwords()
    
    # Load data
    data = load_data()
    if data is None:
        st.error("Failed to load data.")
        return
         
    orders, customers, order_items, order_payments, order_reviews, products, product_translation, geolocation, sellers = data
    
    # Preprocess data
    orders, customers, order_items, order_payments, order_reviews, products, product_translation, geolocation, sellers = preprocess_data(
        orders, customers, order_items, order_payments, order_reviews, products, product_translation, geolocation, sellers
    )

    # Process review data for sentiment analysis
    df = order_reviews[['review_score', 'review_comment_message']].copy()
    df = df.dropna(subset=['review_comment_message'])
    
    # Apply sentiment labeling
    df['sentiment'] = df['review_score'].apply(label_sentiment)
    
    # Apply tokenization 
    df['tokens'] = df['review_comment_message'].apply(lambda x: clean_text(x, stop_words))
    
    # Extract noun + adjective bigrams
    df['noun_adj_bigrams'] = df['tokens'].apply(lambda tokens: extract_noun_adj_bigrams(tokens, nlp))

    # Save to session_state
    st.session_state.df = df

    # KPI Summary
    display_kpi_summary(orders, order_reviews, order_payments)

    # Sidebar
    st.sidebar.title("📋 Navigation")
    menu_option = st.sidebar.selectbox(
        "Choose Analysis Menu:",
        [
            "Menu 1: Delivery Evaluation",
            "Menu 2: Sentiment Analysis", 
            "Menu 3: Market Expansion"
        ]
    )

    # Menu handlers
    if menu_option == "Menu 1: Delivery Evaluation":
        menu_delivery_evaluation(orders, customers, geolocation)
    elif menu_option == "Menu 2: Sentiment Analysis":
        menu_sentiment_analysis(order_reviews, order_items, products, product_translation, nlp, stop_words)
    elif menu_option == "Menu 3: Market Expansion":
        menu_market_expansion(orders, customers, order_reviews)

# Add Kangturu label at bottom left
st.sidebar.markdown("---")
st.sidebar.markdown("Kangturu  \nSSDC2025002")

if __name__ == "__main__":
    main()
