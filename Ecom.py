import streamlit as st
import pandas as pd
import requests
from io import StringIO
import re
from collections import defaultdict
from bisect import bisect_left, bisect_right
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="User Journey Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://172.31.6.70:8123/?database=prod&buffer_size=500000"
USERNAME = "readonly_user"
PASSWORD = "cd267e1a-4db5-4936-a4bb-89f1104ea163"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_data(api_url, query, username, password):
    """Fetches data from the API."""
    try:
        response = requests.post(
            api_url,
            data=query.encode("utf-8"),
            auth=(username, password),
            headers={"Content-Type": "text/plain"}
        )
        if response.status_code == 200:
            return response.text
        else:
            st.error(f"❌ Error: {response.status_code} {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ Connection Error: {str(e)}")
        return None

def parse_data(data):
    """Parses the TSV data into a pandas DataFrame."""
    df = pd.read_csv(
        StringIO(data),
        sep="\t",
        names=['token', 'text', 'start_ts', 'log_date', 'type']
    )
    df['start_ts'] = pd.to_datetime(df['start_ts'])
    return df

def consolidate_klog_sequences(df):
    """Consolidate klog typing sequences into final complete product names."""
    df_sorted = df.sort_values(['token', 'start_ts']).reset_index(drop=True)
    consolidated_products = []
    
    i = 0
    while i < len(df_sorted):
        row = df_sorted.iloc[i]
        
        if row['type'] == 'klog':
            # Start of a potential klog sequence
            token = row['token']
            start_time = row['start_ts']
            klog_sequence = [row]
            
            # Look ahead for more klog entries from same user within time window
            j = i + 1
            time_window = pd.Timedelta(minutes=5)  # 5-minute window for typing sequence
            
            while j < len(df_sorted):
                next_row = df_sorted.iloc[j]
                
                # Stop if different user or too much time passed
                if (next_row['token'] != token or 
                    next_row['start_ts'] - start_time > time_window):
                    break
                
                # If it's another klog entry, add to sequence
                if next_row['type'] == 'klog':
                    klog_sequence.append(next_row)
                    j += 1
                else:
                    # If we hit a non-klog entry, check if it's the final product name
                    if (next_row['type'] in ['category', 'item'] and 
                        is_completion_of_sequence(klog_sequence, next_row)):
                        # Use the category/item name as the final product name
                        final_product = {
                            'token': token,
                            'product_name': clean_product_name(next_row['text']),
                            'search_timestamp': start_time,
                            'final_timestamp': next_row['start_ts'],
                            'search_type': 'klog_sequence',
                            'sequence_length': len(klog_sequence),
                            'added_to_cart': False
                        }
                        consolidated_products.append(final_product)
                        i = j + 1  # Skip past this sequence
                        break
                    else:
                        break
            else:
                # Reached end of data, use longest klog entry as final name
                if klog_sequence:
                    longest_entry = max(klog_sequence, key=lambda x: len(str(x['text'])))
                    final_product = {
                        'token': token,
                        'product_name': clean_product_name(longest_entry['text']),
                        'search_timestamp': start_time,
                        'final_timestamp': longest_entry['start_ts'],
                        'search_type': 'klog_sequence',
                        'sequence_length': len(klog_sequence),
                        'added_to_cart': False
                    }
                    consolidated_products.append(final_product)
                i = j
                continue
            
            # If no category/item found, use the longest klog entry
            if j == i + len(klog_sequence):
                longest_entry = max(klog_sequence, key=lambda x: len(str(x['text'])))
                final_product = {
                    'token': token,
                    'product_name': clean_product_name(longest_entry['text']),
                    'search_timestamp': start_time,
                    'final_timestamp': longest_entry['start_ts'],
                    'search_type': 'klog_sequence',
                    'sequence_length': len(klog_sequence),
                    'added_to_cart': False
                }
                consolidated_products.append(final_product)
                i = j
        
        elif row['type'] in ['category', 'item']:
            # Standalone category/item searches (not part of klog sequence)
            if ('All Categories' not in str(row['text']) and 
                'Shopping Cart' not in str(row['text'])):
                
                final_product = {
                    'token': row['token'],
                    'product_name': clean_product_name(row['text']),
                    'search_timestamp': row['start_ts'],
                    'final_timestamp': row['start_ts'],
                    'search_type': row['type'],
                    'sequence_length': 1,
                    'added_to_cart': False
                }
                consolidated_products.append(final_product)
            i += 1
        else:
            i += 1
    
    return consolidated_products



def is_completion_of_sequence(klog_sequence, category_row):
    """Check if category/item row is the completion of the klog typing sequence."""
    if not klog_sequence:
        return False
    
    # Get the longest klog entry (most complete typing)
    longest_klog = max(klog_sequence, key=lambda x: len(str(x['text'])))
    longest_text = clean_product_name(longest_klog['text']).lower().strip()
    category_text = clean_product_name(category_row['text']).lower().strip()
    
    # Check if category text contains or is similar to the longest klog text
    if longest_text in category_text or category_text.startswith(longest_text[:len(longest_text)//2]):
        return True
    
    return False

def build_user_indices(df):
    """Build optimized indices for faster lookups."""
    user_events = defaultdict(list)
    cart_events = []
    
    # Group events by user and collect cart events
    for idx, row in df.iterrows():
        token = row['token']
        timestamp = row['start_ts']
        event_type = row['type']
        
        user_events[token].append({
            'idx': idx,
            'timestamp': timestamp,
            'type': event_type,
            'text': row['text']
        })
        
        if event_type == 'cart':
            cart_events.append({
                'idx': idx,
                'token': token,
                'timestamp': timestamp
            })
    
    # Sort user events by timestamp for binary search
    for token in user_events:
        user_events[token].sort(key=lambda x: x['timestamp'])
    
    return user_events, cart_events

def binary_search_products(user_events, cart_timestamp, time_window_hours=24):
    """Use binary search to find products within time window."""
    if not user_events:
        return None
    
    timestamps = [event['timestamp'] for event in user_events]
    earliest_time = cart_timestamp - pd.Timedelta(hours=time_window_hours)
    
    # Binary search for the earliest valid timestamp
    left_idx = bisect_left(timestamps, earliest_time)
    right_idx = bisect_left(timestamps, cart_timestamp)
    
    # Search backwards from cart time for the most recent product
    for i in range(right_idx - 1, left_idx - 1, -1):
        event = user_events[i]
        if event['type'] in ['item', 'category', 'klog']:
            # Handle "All Categories" case
            if 'All Categories' in str(event['text']):
                # Look further back for actual product
                for j in range(i - 1, left_idx - 1, -1):
                    prev_event = user_events[j]
                    if (prev_event['type'] in ['item', 'category', 'klog'] and
                        'All Categories' not in str(prev_event['text'])):
                        return prev_event
            else:
                return event
    
    return None

def get_products_added_to_cart_optimized(df, consolidated_products):
    """Optimized version using binary search and hash maps."""
    # Build indices for fast lookups
    user_events, cart_events = build_user_indices(df)
    
    cart_products = []
    processed_carts = set()  # To avoid duplicates
    
    for cart_event in cart_events:
        cart_key = (cart_event['token'], cart_event['timestamp'])
        
        # Skip if already processed (duplicate cart events)
        if cart_key in processed_carts:
            continue
        processed_carts.add(cart_key)
        
        token = cart_event['token']
        cart_timestamp = cart_event['timestamp']
        
        # Get user's events and find the most recent product
        user_event_list = user_events.get(token, [])
        product_event = binary_search_products(user_event_list, cart_timestamp)
        
        if product_event:
            clean_name = clean_product_name(product_event['text'])
            cart_products.append({
                'token': token,
                'product_name': clean_name,
                'cart_timestamp': cart_timestamp
            })
    
    # Update consolidated products to mark which ones were added to cart
    cart_products_set = set()
    for item in cart_products:
        normalized_name = item['product_name'].lower().strip()
        cart_products_set.add((item['token'], normalized_name))
    
    # Mark products as added to cart
    for product in consolidated_products:
        normalized_name = product['product_name'].lower().strip()
        if (product['token'], normalized_name) in cart_products_set:
            product['added_to_cart'] = True
    
    return cart_products

def clean_product_name(text):
    """Extracts clean product name from Amazon text."""
    text = str(text)
    
    # Remove "Amazon.in:" prefix and variations
    text = re.sub(r'^Amazon\.in\s*:\s*', '', text, flags=re.IGNORECASE)
    
    # Remove extra colons and spaces
    text = re.sub(r'^\s*:\s*', '', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_data():
    """Main function to fetch and process data."""
    query = """
    SELECT token, text, start_ts, toDate(start_ts) AS log_date, type
    FROM app_log
    WHERE package = 'in.amazon.mshop.android.shopping'
    ORDER BY start_ts
    """

    with st.spinner("Fetching data from API..."):
        data = fetch_data(API_URL, query, USERNAME, PASSWORD)

    if data:
        with st.spinner("Processing data..."):
            df = parse_data(data)
            if not df.empty:
                # Consolidate klog sequences into final product names
                consolidated_products = consolidate_klog_sequences(df)
                
                # Keep all products without removing duplicates
                final_products = consolidated_products
                
                # Get products added to cart and update consolidated list
                cart_products = get_products_added_to_cart_optimized(df, final_products)
                
                return df, final_products, cart_products
            else:
                st.error("No data returned from the query.")
                return None, None, None
    else:
        return None, None, None

def main():
    # App title and description
    st.title("User Journey Dashboard")
    st.markdown("---")
    
    # Sidebar for filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Fetch and process data
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
        
        df, final_products, cart_products = process_data()
        
        if df is not None and final_products is not None:
            # Convert to DataFrame for easier filtering
            products_df = pd.DataFrame(final_products)
            
            # Token filter
            all_tokens = sorted(products_df['token'].unique())
            selected_tokens = st.multiselect(
                "Select User Tokens",
                options=all_tokens,
                default=all_tokens[:5] if len(all_tokens) > 5 else all_tokens,
                help="Select specific user tokens to analyze"
            )
            
            # Search type filter
            search_types = sorted(products_df['search_type'].unique())
            selected_search_types = st.multiselect(
                "Select Search Types",
                options=search_types,
                default=search_types,
                help="Filter by search type"
            )
            
            # Cart status filter
            cart_status = st.selectbox(
                "Cart Status",
                options=["All", "Added to Cart", "Not Added to Cart"],
                help="Filter by whether products were added to cart"
            )
            
            # Date range filter
            if not products_df.empty:
                min_date = products_df['search_timestamp'].min().date()
                max_date = products_df['search_timestamp'].max().date()
                
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    help="Filter by search date range"
                )
            
            st.markdown("---")
            
            # Apply filters
            filtered_df = products_df.copy()
            
            if selected_tokens:
                filtered_df = filtered_df[filtered_df['token'].isin(selected_tokens)]
            
            if selected_search_types:
                filtered_df = filtered_df[filtered_df['search_type'].isin(selected_search_types)]
            
            if cart_status == "Added to Cart":
                filtered_df = filtered_df[filtered_df['added_to_cart'] == True]
            elif cart_status == "Not Added to Cart":
                filtered_df = filtered_df[filtered_df['added_to_cart'] == False]
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df['search_timestamp'].dt.date >= start_date) &
                    (filtered_df['search_timestamp'].dt.date <= end_date)
                ]
            
            # Display filter summary
            st.info(f"📊 Showing {len(filtered_df)} of {len(products_df)} products")
            
    # Main content area
    if df is not None and final_products is not None:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", len(filtered_df))
        
        with col2:
            cart_count = len(filtered_df[filtered_df['added_to_cart'] == True])
            st.metric("Added to Cart", cart_count)
        
        with col3:
            conversion_rate = (cart_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
        
        with col4:
            unique_users = filtered_df['token'].nunique()
            st.metric("Unique Users", unique_users)
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Product Table", "📊 Analytics", "🛒 Cart Analysis", "📈 Multiple User Journey"])
        
        with tab1:
            st.subheader("Product Search Data")
            
            if not filtered_df.empty:
                # Display options
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_term = st.text_input("🔍 Search products", placeholder="Enter product name to search...")
                with col2:
                    show_all_columns = st.checkbox("Show all columns", value=False)
                
                # Apply search filter
                display_df = filtered_df.copy()
                if search_term:
                    display_df = display_df[
                        display_df['product_name'].str.contains(search_term, case=False, na=False)
                    ]
                
                # Select columns to display
                if show_all_columns:
                    columns_to_show = display_df.columns.tolist()
                else:
                    columns_to_show = ['token', 'product_name', 'search_type', 'search_timestamp', 'added_to_cart']
                
                # Format timestamps for better display
                display_df_formatted = display_df.copy()
                display_df_formatted['search_timestamp'] = display_df_formatted['search_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                if 'final_timestamp' in display_df_formatted.columns:
                    display_df_formatted['final_timestamp'] = display_df_formatted['final_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Display the table
                st.dataframe(
                    display_df_formatted[columns_to_show],
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"product_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No products match the current filters.")
        
        with tab2:
            st.subheader("Product Analytics")
            
            if not filtered_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top products chart
                    top_products = filtered_df['product_name'].value_counts().head(10)
                    if not top_products.empty:
                        fig_products = px.bar(
                            x=top_products.values,
                            y=top_products.index,
                            orientation='h',
                            title="Top 10 Most Searched Products",
                            labels={'x': 'Search Count', 'y': 'Product Name'}
                        )
                        fig_products.update_layout(height=400)
                        st.plotly_chart(fig_products, use_container_width=True)
                
                with col2:
                    # Search type distribution
                    search_type_counts = filtered_df['search_type'].value_counts()
                    fig_types = px.pie(
                        values=search_type_counts.values,
                        names=search_type_counts.index,
                        title="Search Type Distribution"
                    )
                    fig_types.update_layout(height=400)
                    st.plotly_chart(fig_types, use_container_width=True)
                
                # Time series analysis
                st.subheader("Search Activity Over Time")
                filtered_df['date'] = filtered_df['search_timestamp'].dt.date
                daily_searches = filtered_df.groupby('date').size().reset_index(name='count')
                
                fig_timeline = px.line(
                    daily_searches,
                    x='date',
                    y='count',
                    title="Daily Search Activity",
                    labels={'count': 'Number of Searches', 'date': 'Date'}
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        with tab3:
            st.subheader("Cart Analysis")
            
            cart_df = filtered_df[filtered_df['added_to_cart'] == True]
            
            if not cart_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top products added to cart
                    top_cart_products = cart_df['product_name'].value_counts().head(10)
                    fig_cart = px.bar(
                        x=top_cart_products.values,
                        y=top_cart_products.index,
                        orientation='h',
                        title="Top Products Added to Cart",
                        labels={'x': 'Cart Additions', 'y': 'Product Name'}
                    )
                    fig_cart.update_layout(height=400)
                    st.plotly_chart(fig_cart, use_container_width=True)
                
                with col2:
                    # Conversion by search type
                    conversion_by_type = filtered_df.groupby('search_type').agg({
                        'added_to_cart': ['sum', 'count']
                    }).round(2)
                    conversion_by_type.columns = ['Cart_Additions', 'Total_Searches']
                    conversion_by_type['Conversion_Rate'] = (
                        conversion_by_type['Cart_Additions'] / conversion_by_type['Total_Searches'] * 100
                    ).round(1)
                    
                    fig_conversion = px.bar(
                        x=conversion_by_type.index,
                        y=conversion_by_type['Conversion_Rate'],
                        title="Conversion Rate by Search Type",
                        labels={'x': 'Search Type', 'y': 'Conversion Rate (%)'}
                    )
                    fig_conversion.update_layout(height=400)
                    st.plotly_chart(fig_conversion, use_container_width=True)
                
                # Cart products table
                st.subheader("Products Added to Cart")
                cart_display = cart_df[['token', 'product_name', 'search_timestamp', 'search_type']].copy()
                cart_display['search_timestamp'] = cart_display['search_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(cart_display, use_container_width=True)
            else:
                st.info("No products were added to cart with the current filters.")
        
        with tab4:
            st.subheader("Multiple User Journey Analysis")
            
            if not filtered_df.empty:
                # User activity summary
                user_summary = filtered_df.groupby('token').agg({
                    'product_name': 'count',
                    'added_to_cart': 'sum',
                    'search_timestamp': ['min', 'max']
                }).round(2)
                
                user_summary.columns = ['Total_Searches', 'Cart_Additions', 'First_Search', 'Last_Search']
                user_summary['Conversion_Rate'] = (
                    user_summary['Cart_Additions'] / user_summary['Total_Searches'] * 100
                ).round(1)
                
                # Top users by activity
                col1, col2 = st.columns(2)
                
                with col1:
                    top_users_searches = user_summary.nlargest(10, 'Total_Searches')
                    fig_users = px.bar(
                        x=top_users_searches['Total_Searches'],
                        y=top_users_searches.index,
                        orientation='h',
                        title="Top 10 Users by Search Activity",
                        labels={'x': 'Total Searches', 'y': 'User Token'}
                    )
                    fig_users.update_layout(height=400)
                    st.plotly_chart(fig_users, use_container_width=True)
                
                with col2:
                    # User conversion rates
                    users_with_conversions = user_summary[user_summary['Cart_Additions'] > 0]
                    if not users_with_conversions.empty:
                        fig_user_conversion = px.scatter(
                            users_with_conversions,
                            x='Total_Searches',
                            y='Conversion_Rate',
                            size='Cart_Additions',
                            title="User Conversion Analysis",
                            labels={'Total_Searches': 'Total Searches', 'Conversion_Rate': 'Conversion Rate (%)'}
                        )
                        fig_user_conversion.update_layout(height=400)
                        st.plotly_chart(fig_user_conversion, use_container_width=True)
                    else:
                        st.info("No users with cart additions in current filter.")
                
                # User summary table
                st.subheader("User Activity Summary")
                user_display = user_summary.copy()
                user_display['First_Search'] = user_display['First_Search'].dt.strftime('%Y-%m-%d %H:%M:%S')
                user_display['Last_Search'] = user_display['Last_Search'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(user_display, use_container_width=True)
    
    else:
        st.error("Failed to load data. Please check your connection and try again.")
        st.info("Make sure the API endpoint is accessible and credentials are correct.")

if __name__ == "__main__":
    main() 
