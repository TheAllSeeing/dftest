# To adjust in HLS
import colorsys
# To generate cmap, convert names to hex & rgb
from matplotlib.colors import cnames, to_rgb, hex2color, LinearSegmentedColormap


def make_autopct(data):
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
        try:
            base_attr = getattr(base_attr, attr)
        except AttributeError as e:
            raise ValueError(f'Tried to parse invalid function name: {".".join(func_addr)}. No {attr} in {base_attr}! '
                             f'Listed in {base_attr}:\n' + "\n".join(dir(base_attr)))

    return base_attr


def to_hex(color):
    # if type(color) == tuple:
    #     return f"#{}"
    # if color.startswith('#'):
    #     return color
    try:
        return cnames[color]
    except KeyError:
        return color


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
    color_hex = to_hex(color)
    color_hls = colorsys.rgb_to_hls(*to_rgb(color_hex))
    return colorsys.hls_to_rgb(
        color_hls[0],  # Hue remains
        max(0, min(1, amount * color_hls[1])),  #
        color_hls[2]
    )


# From https://stackoverflow.com/a/50230769/10913212
def nonlinear_cmap(step_colors, step_values, name=None):
    """
    Generates a non-linear matplolib color map. colormap meaning a gradient for color coding values
    in a range (e.g for heat map), and non-linear meaning each color takes over a differently-sized
    chunk of the gradient.

    :param step_colors: an ordered list of colors for each "step" on the gradient
    :param step_values: values on the unit interval matching where each step should be on the graient
    :return: a :class:`matplotlib.color.LinearSegmentedColormap` which can be used as a color map for matplotlib and
    seaborn methods
    """
    color_dict = {'red': (), 'green': (), 'blue': ()}
    for step_color, step_value in zip(step_colors, step_values):
        rgb = hex2color(to_hex(step_color))
        color_dict['red'] = color_dict['red'] + ((step_value, rgb[0], rgb[0]),)
        color_dict['green'] = color_dict['green'] + ((step_value, rgb[1], rgb[1]),)
        color_dict['blue'] = color_dict['blue'] + ((step_value, rgb[2], rgb[2]),)
    return LinearSegmentedColormap(name, color_dict)
