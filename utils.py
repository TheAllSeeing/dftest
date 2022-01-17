def pie_autopct(data):
    """
    Generates a function for automatic labeling of pie charts by number and percent values
    :param data: the pie chart data
    :returns: an autolabel function to pass to the "autopct" parameter of pyplot.pie
    """

    def autopct(percent):
        """
        Function for automatic labeling of pie charts by number and percent values
        :param percent: the percent of total a section of a pie chart represents
        :return: a string label, formatted "percent (value)"
        """
        total = sum(data)
        value = int(round(percent * total / 100.0))
        return f'{round(percent, 2)}% ({value})'

    return autopct


# Used for reading config files.
def get_func_from_addr(func_addr: str):
    """
    Returns a function object from its "dotted name",practically, this acts like returning :func:`eval` on
    the input, except it will only work for valid attributes.

    :param func_addr: The dotted name of a funciton, like in an import statement or a line of code:
    e.g module.class.function, module.function, function_name...
    :returns: a function object of the callable the input string refers to.
    """
    func_addr = func_addr.split('.')
    base_attr = __import__(func_addr[0])
    del func_addr[0]
    for attr in func_addr:
        base_attr = getattr(base_attr, attr)
    return base_attr


# From https://stackoverflow.com/a/49601444/10913212
def adjust_lightness(color, amount=0.5):
    """
    adjust the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    # To get RGB from input
    from matplotlib.colors import cnames, to_rgb
    # To adjust in HLS
    import colorsys
    try:
        color_code = cnames[color]
    except KeyError:
        color_code = color
    color_code = colorsys.rgb_to_hls(*to_rgb(color_code))
    return colorsys.hls_to_rgb(
        color_code[0],  # Hue remains
        max(0, min(1, amount * color_code[1])),  #
        color_code[2]
    )
