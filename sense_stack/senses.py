"""
Sense inventories and synonym tables for supported polysemous words.

To add a new word: define its senses here, add synonyms, then generate
training data and train a classifier. See README.md for details.
"""

# Sense definitions: word -> {sense_id: description}
SENSES = {
    'bank': {
        'bank_finance': 'financial institution (deposit money, get loans, ATM, banking)',
        'bank_river': 'edge/shore of a river, lake, or waterway',
        'bank_collection': 'a stored collection or repository (blood bank, food bank, memory bank, data bank)',
    },
    'light': {
        'light_physical': 'physical light: electromagnetic radiation, brightness, a lamp, a source of light, igniting, photons, sunlight, flashlight',
        'light_figurative': 'figurative/metaphorical: not heavy, lightweight, mild, not serious, perspective, way of seeing (in light of, shed light, light reading, light breeze)',
    },
    'plant': {
        'plant_vegetation': 'a living organism (tree, flower, bush, vegetation)',
        'plant_factory': 'a factory, industrial facility, or manufacturing/processing site',
        'plant_verb_grow': 'the act of putting seeds/plants in the ground to grow',
        'plant_verb_place': 'to place something secretly or deliberately (plant evidence, plant a bug, plant a spy)',
    },
    'organ': {
        'organ_body': 'a body part (heart, liver, kidney, organ donor, organ transplant)',
        'organ_instrument': 'a musical instrument (pipe organ, church organ, organ music)',
        'organ_publication': 'an official publication, mouthpiece, or propaganda outlet of an organization',
        'organ_organization': 'a component or department of an organization (organ of government, organ of the state)',
    },
    'star': {
        'star_celestial': 'a celestial body or astronomical object (sun, stars in the sky, neutron star, stargazing)',
        'star_celebrity': 'a famous person, celebrity, or leading performer (movie star, star player, rock star)',
        'star_symbol': 'a star as symbol, shape, rating, or marker (gold star, 5-star rating, Michelin star, star sticker, Star of David)',
    },
}

# Synonyms for substitution scoring: word -> {sense_id: [synonyms]}
# Best synonym (first in list) is used for embedding initialization in fine-tuning.
SYNONYMS = {
    'bank': {
        'bank_finance': ['vault', 'lender', 'savings', 'creditor'],
        'bank_river': ['riverbank', 'shore', 'embankment', 'waterside'],
        'bank_collection': ['reserve', 'stockpile', 'repository', 'depot'],
    },
    'light': {
        'light_physical': ['brightness', 'illumination', 'glow', 'radiance'],
        'light_figurative': ['perspective', 'mild', 'gentle', 'slight'],
    },
    'plant': {
        'plant_vegetation': ['shrub', 'seedling', 'herb', 'sapling'],
        'plant_factory': ['factory', 'facility', 'mill', 'refinery'],
        'plant_verb_grow': ['cultivate', 'sow', 'seed', 'garden'],
        'plant_verb_place': ['smuggle', 'hide', 'stash', 'conceal'],
    },
    'organ': {
        'organ_body': ['kidney', 'liver', 'lung', 'gland'],
        'organ_instrument': ['piano', 'harpsichord', 'synthesizer', 'keyboard'],
        'organ_publication': ['newsletter', 'bulletin', 'gazette', 'journal'],
        'organ_organization': ['department', 'branch', 'division', 'bureau'],
    },
    'star': {
        'star_celestial': ['constellation', 'sun', 'pulsar', 'supernova'],
        'star_celebrity': ['celebrity', 'idol', 'superstar', 'luminary'],
        'star_symbol': ['emblem', 'badge', 'asterisk', 'insignia'],
    },
}

# Convenience: word -> list of sense IDs
WORD_SENSES = {word: list(senses.keys()) for word, senses in SENSES.items()}

# All supported words
SUPPORTED_WORDS = list(SENSES.keys())

# Coarse sense mapping (cascade corpus uses finer senses that map to these)
SENSE_COARSE = {
    'light_illumination': 'light_physical',
    'light_perspective': 'light_figurative',
    'light_modifier': 'light_figurative',
    'light_weight': 'light_figurative',
    'star_shape': 'star_symbol',
    'star_rating': 'star_symbol',
}
