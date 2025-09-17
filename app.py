import streamlit as st
from src.scanner import HealthzScanner

st.set_page_config(page_title="healthz.ca", page_icon="🍁")

st.title("🍁 healthz.ca")
st.subtitle("Canadian Barcode Nutrition Scanner")

scanner = HealthzScanner()

barcode = st.text_input("Enter Barcode/UPC:", placeholder="049000006346")

if st.button("🔍 Scan", type="primary"):
    if barcode:
        with st.spinner("Searching..."):
            product = scanner.scan_barcode(barcode)
        
        if product:
            st.success(f"Found: {product.name}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Brand", product.brand or "N/A")
            with col2:
                st.metric("Category", product.category or "N/A")
            
            if product.nutrition:
                st.subheader("Nutrition Facts")
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Calories", f"{product.nutrition.calories or 0:.0f}")
                with cols[1]:
                    st.metric("Protein", f"{product.nutrition.protein or 0:.1f}g")
                with cols[2]:
                    st.metric("Carbs", f"{product.nutrition.total_carbohydrates or 0:.1f}g")
                
                with st.expander("More Details"):
                    st.write(f"Fat: {product.nutrition.total_fat or 0:.1f}g")
                    st.write(f"Sodium: {product.nutrition.sodium or 0:.1f}mg")
            
            st.caption(f"Source: {product.data_source}")
        else:
            st.error("Product not found")

with st.sidebar:
    st.header("About")
    st.write("healthz.ca helps you make informed nutrition choices by scanning product barcodes.")
    
    st.header("Test Barcodes")
    st.code("049000006346")  # Coca-Cola
    st.code("041318020007")  # Milk
    st.code("038000845512")  # Cereal
