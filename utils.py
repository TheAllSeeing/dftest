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
