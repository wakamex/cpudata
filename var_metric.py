VAR_METHOD = "closed-positive-frontier-v2"


def terminal_crossing(last_price, last_score, slope, intercept):
    """Return where a rising regression crosses the final frontier plateau."""
    last_price = float(last_price)
    last_score = float(last_score)
    slope = float(slope)
    intercept = float(intercept)
    if slope <= 0:
        return None

    crossing = (last_score - intercept) / slope
    return crossing if crossing > last_price else None


def _positive_segment_area(p1, score1, p2, score2, slope, intercept):
    """Integrate the positive gap between two linear segments."""
    height1 = score1 - (slope * p1 + intercept)
    height2 = score2 - (slope * p2 + intercept)

    if height1 <= 0 and height2 <= 0:
        return 0.0
    if height1 >= 0 and height2 >= 0:
        return (p2 - p1) * (height1 + height2) / 2

    crossing = p1 + (p2 - p1) * -height1 / (height2 - height1)
    if height1 > 0:
        return (crossing - p1) * height1 / 2
    return (p2 - crossing) * height2 / 2


def calc_auc_above_regression(frontier_prices, frontier_scores, slope, intercept):
    """Calculate the closed area where a frontier is above the regression.

    The observed frontier is piecewise linear. Its final score is extended
    horizontally until the rising regression catches it, closing the region.
    """
    slope = float(slope)
    intercept = float(intercept)
    points = [(float(price), float(score)) for price, score in zip(frontier_prices, frontier_scores)]
    if len(points) < 2:
        return 0.0

    total_area = sum(
        _positive_segment_area(p1, s1, p2, s2, slope, intercept)
        for (p1, s1), (p2, s2) in zip(points, points[1:])
    )

    last_price, last_score = points[-1]
    crossing = terminal_crossing(last_price, last_score, slope, intercept)
    if crossing is not None:
        total_area += _positive_segment_area(
            last_price,
            last_score,
            crossing,
            last_score,
            slope,
            intercept,
        )

    return total_area
