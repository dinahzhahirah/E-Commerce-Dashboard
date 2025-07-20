import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
from datetime import datetime
import statsmodels.api as sm
import seaborn as sns
from wordcloud import WordCloud
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
from collections import Counter
import string
import warnings
warnings.filterwarnings('ignore')
import gdown
import zipfile
import os

# Unduh file dari Google Drive
url = "https://drive.google.com/drive/folders/1DYalJ1kEVPClMaEgt8kzgrzUNa7ZGQ2F?usp=sharing"
output = "data.zip"
gdown.download(url, output, quiet=False)

# Ekstrak zip
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("data")

# Load spaCy Portuguese model
@st.cache_resource
def load_spacy_model():
    """Load spaCy Portuguese model"""
    try:
        nlp = spacy.load("pt_core_news_sm")
        return nlp
    except OSError:
        st.error("Portuguese spaCy model not found. Please install it with: python -m spacy download pt_core_news_sm")
        return None

# Get Portuguese stopwords
@st.cache_data
def get_portuguese_stopwords():
    """Get Portuguese stopwords from spaCy"""
    nlp = spacy.load("pt_core_news_sm")
    return nlp.Defaults.stop_words

# Page config
st.set_page_config(
    page_title="E-commerce Analytics Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        orders = pd.read_csv('data/orders_dataset.csv')
        customers = pd.read_csv('data/customers_dataset.csv')
        order_items = pd.read_csv('data/order_items_dataset.csv')
        order_payments = pd.read_csv('data/order_payments_dataset.csv')
        order_reviews = pd.read_csv('data/order_reviews_dataset.csv')
        products = pd.read_csv('data/products_dataset.csv')
        product_translation = pd.read_csv('data/product_category_name_translation.csv')
        geolocation = pd.read_csv('data/geolocation_dataset.csv')
        sellers = pd.read_csv('data/sellers_dataset.csv')

        return orders, customers, order_items, order_payments, order_reviews, products, product_translation, geolocation, sellers
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(orders, customers, order_items, order_payments, order_reviews, products, product_translation, geolocation, sellers):
    """Preprocess data for analysis"""
    
    # Convert datetime columns
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')
    orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'], errors='coerce')
    orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'], errors='coerce')
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'], errors='coerce')
    orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'], errors='coerce')
    
    # Calculate delivery metrics
    orders['actual_delivery_days'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
    orders['estimated_delivery_days'] = (orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']).dt.days
    orders['delivery_delay'] = orders['actual_delivery_days'] - orders['estimated_delivery_days']
    orders['is_delayed'] = orders['delivery_delay'] > 0
    
    # Clean review text
    order_reviews['review_comment_message'] = order_reviews['review_comment_message'].fillna('')
    order_reviews['review_comment_title'] = order_reviews['review_comment_title'].fillna('')
    
    return orders, customers, order_items, order_payments, order_reviews, products, product_translation, geolocation, sellers

# LABEL SENTIMEN BERDASARKAN SKOR ULASAN
def label_sentiment(score):
    if score >= 4:
        return 'Positive'
    elif score <= 2:
        return 'Negative'
    else:
        return 'Neutral'

# Fungsi pembersih teks
def clean_text(text):
    stopwords = get_portuguese_stopwords()
    
    # Hilangkan angka dan tanda baca
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Lowercase dan split (tokenisasi manual)
    tokens = text.lower().split()
    
    # Buang stopwords
    tokens = [word for word in tokens if word not in stopwords]
    
    return tokens

# BIGRAM: NOUN + ADJECTIVE WITH SPACY
nlp = spacy.load("pt_core_news_sm")

def extract_noun_adj_bigrams(tokens):
    """
    Ambil bigram (dua kata berurutan) yang terdiri dari noun + adjective dari daftar token.
    Gunakan spaCy untuk tag POS.
    """
    text = " ".join(tokens)
    doc = nlp(text)
    bigrams = []

    for i in range(len(doc) - 1):
        token1 = doc[i]
        token2 = doc[i + 1]

        # Kalau token pertama adalah noun/proper noun dan kedua adalah adjective
        if token1.pos_ in ["NOUN", "PROPN"] and token2.pos_ == "ADJ":
            bigrams.append(f"{token1.text} {token2.text}")

    return bigrams
    
# KPI Summary
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

    # Bar chart di bawah map
    fig_delay = px.bar(
        state_metrics_geo.sort_values('Avg_Delay_Days', ascending=True).head(15).iloc[::-1],
        x='Avg_Delay_Days',
        y='State_Full',
        orientation='h',
        color='Avg_Delay_Days',
        color_continuous_scale='Reds',
        title='Top 15 States by Average Delivery Delay',
        labels={
            'State_Full': 'State',
            'Avg_Delay_Days': 'Average Delivery Delay (days)'
        },
        range_color=[-20, -10],
        text='Avg_Delay_Days'
    )

    fig_delay.update_layout(
        height=500,
        margin=dict(l=120),
        coloraxis_colorbar=dict(
            title='Average Delivery Delay',
            tickvals=[-20, -18, -16, -14, -12, -10][::-1],
            ticktext=[str(i) for i in [-20, -18, -16, -14, -12, -10][::-1]]
        )
    )
    fig_delay.update_traces(
        marker=dict(line=dict(width=0)),
        textposition='auto',
        texttemplate='%{text:.2f}',
        textfont_size=12
    )
    fig_delay.update_coloraxes(reversescale=True)
    st.plotly_chart(fig_delay, use_container_width=True)

    # Insights
    st.subheader("🔍 Key Insights:")

    problematic_states = state_metrics[state_metrics['Avg_Delay_Days'] > 3].head(5)
    high_volume_delayed = state_metrics[(state_metrics['Total_Orders'] > 1000) & (state_metrics['Avg_Delay_Days'] > 2)].head(3)

    insights = []
    if len(problematic_states) > 0:
        states_list = ", ".join(problematic_states['State'].tolist())
        insights.append(f"🚨 Provinsi dengan keterlambatan > 3 hari: {states_list} → Evaluasi mitra logistik lokal")

    if len(high_volume_delayed) > 0:
        hv_states = ", ".join(high_volume_delayed['State'].tolist())
        insights.append(f"⚠️ Volume pesanan tinggi dengan keterlambatan: {hv_states} → Risiko churn pelanggan")

    best_performers = state_metrics[state_metrics['Avg_Delay_Days'] < 1].head(3)
    if len(best_performers) > 0:
        best_states = ", ".join(best_performers['State'].tolist())
        insights.append(f"✅ Provinsi dengan performa bagus: {best_states} → Referensi praktik terbaik")

    for insight in insights:
        st.write(f"• {insight}")

    st.subheader("🎯 Dampak Bisnis:")
    st.write("• **Langsung mengurangi keluhan pelanggan**, meningkatkan loyalitas")
    st.write("• **Optimalkan biaya logistik**, alokasikan ulang sumber daya ke wilayah bermasalah")

def menu_sentiment_analysis(order_reviews, order_items, products, product_translation, nlp, stop_words):
    st.header("MENU 2: Analisis Sentimen Aspek 'Pengiriman' dan 'Produk'")
    
    # Get preprocessed data from session state
    if 'df' not in st.session_state:
        st.error("Data belum diproses. Silakan refresh halaman.")
        return
    
    df = st.session_state.df
    
    st.subheader("📊 Sentiment Analysis Overview")
    
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
        # Sentiment bar chart
        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title='Number of Reviews by Sentiment',
            color=sentiment_counts.index,
            color_discrete_map={
                'Positive': 'green',
                'Neutral': 'yellow',
                'Negative': 'red'
            }
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
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
    category_metrics['review_volume_ratio'] = category_metrics['Avg_Review'] / (category_metrics['Total_Sales'] / 1000)
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
        st.subheader("🎯 Market Segment Distribution")
        
        segment_dist = state_analysis['Market_Segment'].value_counts()
        
        fig_pie = px.pie(
            values=segment_dist.values,
            names=segment_dist.index,
            title='States by Market Segment',
            color_discrete_map={
                'Strong Market': 'green',
                'Expansion Target': 'gold',
                'Volume Leader': 'blue',
                'Evaluate/Leave': 'red'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("📈 Top Expansion Targets")
        expansion_targets = state_analysis[state_analysis['Market_Segment'] == 'Expansion Target'].sort_values('Avg_Review_Score', ascending=False)
        
        if len(expansion_targets) > 0:
            fig_targets = px.bar(
                expansion_targets.head(10),
                x='Avg_Review_Score',
                y='State',
                orientation='h',
                color='Order_Count',
                title='Top Expansion Target States',
                color_continuous_scale='Blues'
            )
            fig_targets.update_layout(height=400)
            st.plotly_chart(fig_targets, use_container_width=True)
        else:
            st.write("Tidak ada target ekspansi yang teridentifikasi.")
    
    # Detailed analysis
    st.subheader("🌍 Detailed Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Strong Markets (High Score - High Volume)")
        strong_markets = state_analysis[state_analysis['Market_Segment'] == 'Strong Market'].sort_values('Order_Count', ascending=False)
        if len(strong_markets) > 0:
            st.dataframe(strong_markets[['State', 'Order_Count', 'Avg_Review_Score']].head(10))
        else:
            st.write("No strong markets identified.")
    
    with col2:
        st.subheader("🟡 Expansion Targets (High Score - Low Volume)")
        expansion_targets_detailed = state_analysis[state_analysis['Market_Segment'] == 'Expansion Target'].sort_values('Avg_Review_Score', ascending=False)
        if len(expansion_targets_detailed) > 0:
            st.dataframe(expansion_targets_detailed[['State', 'Order_Count', 'Avg_Review_Score']].head(10))
        else:
            st.write("No expansion targets identified.")
    
    # Insights
    st.subheader("🔍 Key Insights:")
    
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

    # Process review data for sentiment analysis - FIXED TO USE RegexpTokenizer
    df = order_reviews[['review_score', 'review_comment_message']].copy()
    df = df.dropna(subset=['review_comment_message'])
    
    # Apply sentiment labeling
    df['sentiment'] = df['review_score'].apply(label_sentiment)
    
    # Apply tokenization using RegexpTokenizer
    df['tokens'] = df['review_comment_message'].apply(clean_text)
    
    # Extract noun + adjective bigrams
    df['noun_adj_bigrams'] = df['tokens'].apply(extract_noun_adj_bigrams)

    # Simpan ke session_state atau global jika ingin dipakai lintas fungsi:
    st.session_state.df = df  # <- atau pakai global df

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

    # Menu handler
    if menu_option == "🔥 Menu 1: Evaluasi Pengiriman":
        menu_delivery_evaluation(orders, customers, geolocation)
    elif menu_option == "🔥 Menu 2: Analisis Sentimen":
        menu_sentiment_analysis(order_reviews, order_items, products, product_translation)
    elif menu_option == "🔥 Menu 3: Peluang Ekspansi Pasar":
        menu_market_expansion(orders, customers, order_reviews)

if __name__ == "__main__":
    main()
