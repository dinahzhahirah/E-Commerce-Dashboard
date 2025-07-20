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
    """Load all datasets from Google Drive"""
    try:
        # Check if data directory exists, if not create it and download
        if not os.path.exists('data') or not os.listdir('data'):
            st.info("Downloading data files from Google Drive...")
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Download individual files from Google Drive
            # Note: You need to make each file publicly accessible and get the file ID from sharing URL
            files_to_download = {
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
            
            # Download each file
            for filename, file_id in files_to_download.items():
                try:
                    if file_id != 'REPLACE_WITH_ACTUAL_FILE_ID':  # Only download if ID is provided
                        url = f"https://drive.google.com/uc?id={file_id}"
                        output_path = f"data/{filename}"
                        gdown.download(url, output_path, quiet=False)
                        st.success(f"Downloaded {filename}")
                    else:
                        st.warning(f"File ID not provided for {filename}")
                except Exception as e:
                    st.error(f"Failed to download {filename}: {e}")
            
            st.success("Data download completed!")

        # Load datasets
        datasets = {}
        required_files = [
            'customers_dataset.csv',
            'geolocation_dataset.csv', 
            'order_items_dataset.csv',
            'order_payments_dataset.csv',
            'order_reviews_dataset.csv',
            'orders_dataset.csv',
            'product_category_name_translation.csv',
            'products_dataset.csv',
            'sellers_dataset.csv'
        ]
        
        # Check if all required files exist
        missing_files = []
        for filename in required_files:
            filepath = f'data/{filename}'
            if not os.path.exists(filepath):
                missing_files.append(filename)
        
        if missing_files:
            st.error(f"Missing required files: {', '.join(missing_files)}")
            st.info("Please download the files manually and place them in the 'data' directory, or provide the correct Google Drive file IDs.")
            return None
        
        # Load all datasets
        st.info("Loading datasets...")
        customers = pd.read_csv('data/customers_dataset.csv')
        geolocation = pd.read_csv('data/geolocation_dataset.csv')
        order_items = pd.read_csv('data/order_items_dataset.csv')
        order_payments = pd.read_csv('data/order_payments_dataset.csv')
        order_reviews = pd.read_csv('data/order_reviews_dataset.csv')
        orders = pd.read_csv('data/orders_dataset.csv')
        product_translation = pd.read_csv('data/product_category_name_translation.csv')
        products = pd.read_csv('data/products_dataset.csv')
        sellers = pd.read_csv('data/sellers_dataset.csv')
        
        st.success("All datasets loaded successfully!")
        
        return orders, customers, order_items, order_payments, order_reviews, products, product_translation, geolocation, sellers
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please make sure all data files are in the 'data' directory")
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
        text = " ".join(tokens)
        doc = nlp(text)
        bigrams = []

        for i in range(len(doc) - 1):
            token1 = doc[i]
            token2 = doc[i + 1]

            # Check if first token is noun and second is adjective
            if token1.pos_ in ["NOUN", "PROPN"] and token2.pos_ == "ADJ":
                bigrams.append(f"{token1.text} {token2.text}")

        return bigrams
    except:
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
    
    st.subheader("Sentiment Analysis Overview")
    
    # Display basic sentiment statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        fig_sentiment = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Sentiment Distribution',
            color_discrete_map={
                'Positive': 'green',
                'Neutral': 'yellow',
                'Negative': 'red'
            }
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Aspect-based sentiment analysis
        aspek_dict = {
            "Buyer Preferences": ["gostei", "não gostei", "amei", "odiei", "adoro", "detesto"],
            "Product Quality": ["produto", "qualidade", "defeito", "bom", "ruim", "quebrado"],
            "Delivery Performance": ["entrega", "prazo", "transportadora", "rápido", "atrasado"]
        }

        def klasifikasi_sentimen(text):
            if pd.isna(text):
                return {"Buyer Preferences": 0, "Product Quality": 0, "Delivery Performance": 0}
            
            text = str(text).lower()
            hasil = {"Buyer Preferences": 0, "Product Quality": 0, "Delivery Performance": 0}
            
            for aspek, kata_kunci in aspek_dict.items():
                for kata in kata_kunci:
                    if kata in text:
                        hasil[aspek] += 1
            return hasil

        # Process ABSA
        df["aspek_sentimen"] = df["review_comment_message"].apply(klasifikasi_sentimen)

        # Calculate aspect counts
        aspect_counts = {
            "Buyer Preferences": 0,
            "Product Quality": 0,
            "Delivery Performance": 0
        }

        for hasil in df["aspek_sentimen"]:
            for aspek, count in hasil.items():
                aspect_counts[aspek] += count

        # Create bar chart
        fig_aspects = px.bar(
            x=list(aspect_counts.keys()),
            y=list(aspect_counts.values()),
            title='Aspect-Based Sentiment Distribution',
            labels={'x': 'Aspect', 'y': 'Keyword Frequency'},
            color=list(aspect_counts.values()),
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_aspects, use_container_width=True)
    
    # Bigram Analysis
    st.subheader("🔍 Bigram Analysis (Noun + Adjective)")
    
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
    st.subheader("🛒 Product Performance Analysis")
    
    # Merge product data
    product_reviews = order_items.merge(order_reviews, on='order_id', how='left')
    product_reviews = product_reviews.merge(products, on='product_id', how='left')
    product_reviews = product_reviews.merge(product_translation, on='product_category_name', how='left')
    
    # Calculate metrics by category
    category_metrics = product_reviews.groupby('product_category_name_english').agg({
        'order_id': 'count',
        'review_score': 'mean'
    }).reset_index()
    category_metrics.columns = ['Category', 'Total_Sales', 'Avg_Review']
    category_metrics = category_metrics.dropna().sort_values('Total_Sales', ascending=False).head(15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Top Categories by Sales Volume")
        fig_sales = px.bar(
            category_metrics.head(10),
            x='Total_Sales',
            y='Category',
            orientation='h',
            title='Top 10 Categories by Sales'
        )
        fig_sales.update_layout(height=400)
        st.plotly_chart(fig_sales, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Top Categories by Review Score")
        fig_review = px.bar(
            category_metrics.sort_values('Avg_Review', ascending=False).head(10),
            x='Avg_Review',
            y='Category',
            orientation='h',
            color='Avg_Review',
            color_continuous_scale='Greens',
            title='Top 10 Categories by Review Score'
        )
        fig_review.update_layout(height=400)
        st.plotly_chart(fig_review, use_container_width=True)
    
    # Insights
    st.subheader("🔍 Key Insights:")
    
    # Find categories with high volume but low reviews
    problematic_categories = category_metrics[
        (category_metrics['Total_Sales'] > category_metrics['Total_Sales'].quantile(0.7)) & 
        (category_metrics['Avg_Review'] < category_metrics['Avg_Review'].median())
    ].head(3)
    
    high_potential = category_metrics[
        (category_metrics['Avg_Review'] > category_metrics['Avg_Review'].quantile(0.8)) & 
        (category_metrics['Total_Sales'] < category_metrics['Total_Sales'].median())
    ].head(3)
    
    insights = []
    if len(problematic_categories) > 0:
        prob_cats = ", ".join(problematic_categories['Category'].tolist())
        insights.append(f"⚠️ Kategori volume tinggi tapi review rendah: {prob_cats} → Indikator overpromising")
    
    if len(high_potential) > 0:
        pot_cats = ", ".join(high_potential['Category'].tolist())
        insights.append(f"🚀 Kategori review tinggi tapi volume rendah: {pot_cats} → Potensi boost campaign")
    
    for insight in insights:
        st.write(f"• {insight}")
    
    st.subheader("🎯 Dampak Bisnis:")
    st.write("• **Langsung meningkatkan konversi dan kepuasan**")
    st.write("• **Optimalkan alokasi iklan untuk produk berkualitas**")

def menu_market_expansion(orders, customers, order_reviews):
    """Menu 3: Market Expansion Opportunities"""
    st.header("🔥 MENU 3: Identifikasi Peluang Ekspansi Pasar")
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Market Opportunity Matrix")
        
        # Scatter plot
        fig_scatter = px.scatter(
            state_analysis,
            x='Avg_Review_Score',
            y='Order_Count',
            color='Market_Segment',
            size='Order_Count',
            hover_data=['State'],
            title='Market Expansion Opportunities',
            labels={
                'Avg_Review_Score': 'Average Review Score',
                'Order_Count': 'Number of Orders'
            },
            color_discrete_map={
                'Strong Market': 'green',
                'Expansion Target': 'gold',
                'Volume Leader': 'blue',
                'Evaluate/Leave': 'red'
            }
        )
        
        # Add quadrant lines
        fig_scatter.add_hline(y=median_orders, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_vline(x=median_review, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
            st.subheader("🟡 Expansion Targets (High Score - Low Volume)")
            expansion_targets_detailed = state_analysis[state_analysis['Market_Segment'] == 'Expansion Target'].sort_values('Avg_Review_Score', ascending=False)
            if len(expansion_targets_detailed) > 0:
                st.dataframe(expansion_targets_detailed[['State', 'Order_Count', 'Avg_Review_Score']].head(10))
            else:
                st.write("No expansion targets identified.")
        
    # Insights
    st.subheader("🔍 Key Insights:")
        
    # PERBAIKAN 1: Definisikan strong_markets
    strong_markets = state_analysis[state_analysis['Market_Segment'] == 'Strong Market']
    
    insights = []
    if len(expansion_targets_detailed) > 0:
        top_targets = expansion_targets_detailed.head(3)['State'].tolist()
        avg_score = expansion_targets_detailed.head(3)['Avg_Review_Score'].mean()
        insights.append(f"🎯 Top expansion targets: {', '.join(top_targets)} dengan rata-rata review {avg_score:.2f}")
        insights.append(f"📈 Wilayah ini memiliki pengalaman pengguna yang baik, tinggal ditingkatkan visibilitasnya")
    
    if len(strong_markets) > 0:
        strong_count = len(strong_markets)
        insights.append(f"💪 {strong_count} provinsi sudah menjadi pasar kuat dengan performa tinggi")
    
    for insight in insights:
        st.write(f"• {insight}")
    
    st.subheader("🎯 Dampak Bisnis:")
    st.write("• Perusahaan bisa **bertumbuh lebih cepat dengan risiko lebih rendah**")
    st.write("• Ekspansi diarahkan ke **wilayah yang sudah terbukti punya respons positif**")
    st.write("• Bisa memprioritaskan **alokasi iklan dan logistik ke wilayah potensial**")

# Main app
def main():
    st.title("E-commerce Analytics Dashboard")
    st.markdown("### Analisis Komprehensif untuk Optimasi Bisnis E-commerce")
    
    # Load spaCy model and stopwords
    nlp = load_spacy_model()
    if nlp is None:
        st.error("Failed to load spaCy model. Please install Portuguese model first.")
        return
    
    stop_words = get_portuguese_stopwords()
    
    # Load data
    data = load_data()
    if data is None:
        st.error("Failed to load data. Please check if data files are in the correct location.")
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
    
    # PERBAIKAN 3: Extract noun + adjective bigrams dengan lambda
    df['noun_adj_bigrams'] = df['tokens'].apply(lambda tokens: extract_noun_adj_bigrams(tokens, nlp))

    # Simpan ke session_state
    st.session_state.df = df

    # KPI Summary
    display_kpi_summary(orders, order_reviews, order_payments)

    # Sidebar
    st.sidebar.title("📋 Navigation")
    menu_option = st.sidebar.selectbox(
        "Pilih Menu Analisis:",
        [
            "🔥 Menu 1: Evaluasi Pengiriman",
            "🔥 Menu 2: Analisis Sentimen",
            "🔥 Menu 3: Peluang Ekspansi Pasar"
        ]
    )

    # PERBAIKAN 2: Menu handler dengan parameter yang benar
    if menu_option == "🔥 Menu 1: Evaluasi Pengiriman":
        menu_delivery_evaluation(orders, customers, geolocation)
    elif menu_option == "🔥 Menu 2: Analisis Sentimen":
        menu_sentiment_analysis(order_reviews, order_items, products, product_translation, nlp, stop_words)
    elif menu_option == "🔥 Menu 3: Peluang Ekspansi Pasar":
        menu_market_expansion(orders, customers, order_reviews)

if __name__ == "__main__":
    main()
