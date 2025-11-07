import chainlit as cl

# Define a custom function to query a product database
def get_product_info(product_id: str):
    # Simulate fetching product information
    return {
        "id": product_id,
        "name": "Sample Product",
        "price": 19.99,
        "description": "This is a sample product."
    }

# Register the function with Chainlit
@cl.function(name="get_product_info", description="Fetches product information for a given product ID.")
def product_info_function(product_id: str):
    return get_product_info(product_id)