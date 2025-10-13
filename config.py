from torchvision import transforms

data_config = {
    'amazon': {
        'splits': [
            ['bicycle_helmet', 'hand_truck', 'jacket', 'duffel_bag', 'cooler', 'binoculars', 'bicycle'],
            ['purse', 'skateboard', 'suitcase', 'tire_wheel', 'mobile_phone', 'hat', 'shoes'],
            ['backpack', 'portable_speaker', 'stroller', 'beverage_bottle', 'food_container', 'box', 'cart'],
            ['headphones', 'trash_can', 'poster_tube', 'pet_carrier', 'musical_instrument', 'book', 'tackle_box'],
            ['sports_equipment', 'portable_chair', 'sports_ball', 'bucket', 'umbrella', 'hardshell_case']
        ],
        'classes': ['tackle_box', 'portable_chair', 'bicycle', 'poster_tube', 'duffel_bag', 'sports_equipment', 'hat', 'box', 'purse', 'hardshell_case', 'suitcase', 'umbrella', 'musical_instrument', 'trash_can', 'hand_truck', 'pet_carrier', 'sports_ball', 'mobile_phone', 'portable_speaker', 'binoculars', 'headphones', 'beverage_bottle', 'tire_wheel', 'jacket', 'cooler', 'book', 'stroller', 'backpack', 'food_container', 'cart', 'bucket', 'shoes', 'bicycle_helmet', 'skateboard'],
        'root': './groundingdino_cropped',
        'transform': None,
    }
}