SPLIT_POINT_MIN_DISTANCE = 5
MINIMAL_PIXEL_COUNT = 100
split_count_per_x_range = 1/8 #(1 split point per 8 pixels)
def evaluate_character_combinations(word_image, splitpoints):

    for xval, val in splitpoints:
        print(xval)

    # Prepare characters for evaluation

    """
    RULE BASED APPROACH

    Distance in between split points
    Total pixel amount between split points
    Split count / x-range
    character_recognition feedback

    """

    """
    count pixels between splits, order by most probable unnessecary
        => fitness

    close to two other split splitpoints
        => fitness minus'

    Check if splits are closer thatn MIN_DISTANCE
        remove which one?

    """
