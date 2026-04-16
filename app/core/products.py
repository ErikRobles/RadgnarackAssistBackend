from app.schemas.product import ProductProfile

PRODUCT_CATALOG = [
    ProductProfile(
        product_id="rad-1", name="Rad-1 Light", max_bikes=2, 
        max_weight_per_bike=35.0, supported_hitch_sizes=[1.25, 2.0],
        max_tire_width=2.5, supports_step_through=True, 
        is_ebike_rated=False, extension_clearance_inches=4.0, base_price=299.0
    ),
    ProductProfile(
        product_id="hitch-pro", name="Hitch Pro HD", max_bikes=4, 
        max_weight_per_bike=60.0, supported_hitch_sizes=[2.0],
        max_tire_width=5.0, supports_step_through=True, 
        is_ebike_rated=True, extension_clearance_inches=8.0, base_price=599.0
    ),
    ProductProfile(
        product_id="e-rack-2", name="E-Rack 2", max_bikes=2, 
        max_weight_per_bike=65.0, supported_hitch_sizes=[2.0],
        max_tire_width=3.0, supports_step_through=True, 
        is_ebike_rated=True, extension_clearance_inches=6.0, base_price=499.0
    )
]
