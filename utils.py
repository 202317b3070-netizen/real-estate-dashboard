def format_indian_price(price):
    if price >= 10000000:
        return f"₹ {price/10000000:.2f} Cr"
    elif price >= 100000:
        return f"₹ {price/100000:.2f} L"
    else:
        return f"₹ {price:,.0f}"
 
