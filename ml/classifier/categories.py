CATEGORIES = (
    "acoustic_guitar",
    "airplane",
    "applause",
    "bird",
    "car",
    "cat",
    "child",
    "church_bell",
    # "crowd",  # commented out because some of these contain laughter
    "dog_barking",
    "engine",
    "fireworks",
    "footstep",
    "glass_breaking",
    "hammer",
    "helicopter",
    "knock",
    "laughter",
    "mouse_click",
    "ocean_surf",
    "rustle",
    "scream",
    "speech_fs",
    "squeak",
    "tone",
    "violin",
    "water_tap",
    "whistle",
)
NON_LAUGHTER_CATEGORIES = tuple(set(CATEGORIES) - {"laughter"})


def is_laughter_category(category_name):
    return category_name == "laughter"
